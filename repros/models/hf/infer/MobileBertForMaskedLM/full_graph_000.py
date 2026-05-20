class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[256, 128]", arg1_1: "i64[1, 512]", arg2_1: "f32[30522, 128]", arg3_1: "f32[512, 384]", arg4_1: "f32[512]", arg5_1: "f32[512, 512]", arg6_1: "f32[2, 512]", arg7_1: "f32[512]", arg8_1: "f32[512]", arg9_1: "f32[128, 512]", arg10_1: "f32[128]", arg11_1: "f32[128]", arg12_1: "f32[128]", arg13_1: "f32[128, 512]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[128]", arg17_1: "f32[128, 128]", arg18_1: "f32[128]", arg19_1: "f32[128, 128]", arg20_1: "f32[128]", arg21_1: "f32[128, 512]", arg22_1: "f32[128]", arg23_1: "f32[128, 128]", arg24_1: "f32[128]", arg25_1: "f32[128]", arg26_1: "f32[128]", arg27_1: "f32[512, 128]", arg28_1: "f32[512]", arg29_1: "f32[128, 512]", arg30_1: "f32[128]", arg31_1: "f32[128]", arg32_1: "f32[128]", arg33_1: "f32[512, 128]", arg34_1: "f32[512]", arg35_1: "f32[128, 512]", arg36_1: "f32[128]", arg37_1: "f32[128]", arg38_1: "f32[128]", arg39_1: "f32[512, 128]", arg40_1: "f32[512]", arg41_1: "f32[128, 512]", arg42_1: "f32[128]", arg43_1: "f32[128]", arg44_1: "f32[128]", arg45_1: "f32[512, 128]", arg46_1: "f32[512]", arg47_1: "f32[128, 512]", arg48_1: "f32[128]", arg49_1: "f32[128]", arg50_1: "f32[128]", arg51_1: "f32[512, 128]", arg52_1: "f32[512]", arg53_1: "f32[512]", arg54_1: "f32[512]", arg55_1: "f32[128, 512]", arg56_1: "f32[128]", arg57_1: "f32[128]", arg58_1: "f32[128]", arg59_1: "f32[128, 512]", arg60_1: "f32[128]", arg61_1: "f32[128]", arg62_1: "f32[128]", arg63_1: "f32[128, 128]", arg64_1: "f32[128]", arg65_1: "f32[128, 128]", arg66_1: "f32[128]", arg67_1: "f32[128, 512]", arg68_1: "f32[128]", arg69_1: "f32[128, 128]", arg70_1: "f32[128]", arg71_1: "f32[128]", arg72_1: "f32[128]", arg73_1: "f32[512, 128]", arg74_1: "f32[512]", arg75_1: "f32[128, 512]", arg76_1: "f32[128]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[512, 128]", arg80_1: "f32[512]", arg81_1: "f32[128, 512]", arg82_1: "f32[128]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "f32[512, 128]", arg86_1: "f32[512]", arg87_1: "f32[128, 512]", arg88_1: "f32[128]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "f32[512, 128]", arg92_1: "f32[512]", arg93_1: "f32[128, 512]", arg94_1: "f32[128]", arg95_1: "f32[128]", arg96_1: "f32[128]", arg97_1: "f32[512, 128]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[512]", arg101_1: "f32[128, 512]", arg102_1: "f32[128]", arg103_1: "f32[128]", arg104_1: "f32[128]", arg105_1: "f32[128, 512]", arg106_1: "f32[128]", arg107_1: "f32[128]", arg108_1: "f32[128]", arg109_1: "f32[128, 128]", arg110_1: "f32[128]", arg111_1: "f32[128, 128]", arg112_1: "f32[128]", arg113_1: "f32[128, 512]", arg114_1: "f32[128]", arg115_1: "f32[128, 128]", arg116_1: "f32[128]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[512, 128]", arg120_1: "f32[512]", arg121_1: "f32[128, 512]", arg122_1: "f32[128]", arg123_1: "f32[128]", arg124_1: "f32[128]", arg125_1: "f32[512, 128]", arg126_1: "f32[512]", arg127_1: "f32[128, 512]", arg128_1: "f32[128]", arg129_1: "f32[128]", arg130_1: "f32[128]", arg131_1: "f32[512, 128]", arg132_1: "f32[512]", arg133_1: "f32[128, 512]", arg134_1: "f32[128]", arg135_1: "f32[128]", arg136_1: "f32[128]", arg137_1: "f32[512, 128]", arg138_1: "f32[512]", arg139_1: "f32[128, 512]", arg140_1: "f32[128]", arg141_1: "f32[128]", arg142_1: "f32[128]", arg143_1: "f32[512, 128]", arg144_1: "f32[512]", arg145_1: "f32[512]", arg146_1: "f32[512]", arg147_1: "f32[128, 512]", arg148_1: "f32[128]", arg149_1: "f32[128]", arg150_1: "f32[128]", arg151_1: "f32[128, 512]", arg152_1: "f32[128]", arg153_1: "f32[128]", arg154_1: "f32[128]", arg155_1: "f32[128, 128]", arg156_1: "f32[128]", arg157_1: "f32[128, 128]", arg158_1: "f32[128]", arg159_1: "f32[128, 512]", arg160_1: "f32[128]", arg161_1: "f32[128, 128]", arg162_1: "f32[128]", arg163_1: "f32[128]", arg164_1: "f32[128]", arg165_1: "f32[512, 128]", arg166_1: "f32[512]", arg167_1: "f32[128, 512]", arg168_1: "f32[128]", arg169_1: "f32[128]", arg170_1: "f32[128]", arg171_1: "f32[512, 128]", arg172_1: "f32[512]", arg173_1: "f32[128, 512]", arg174_1: "f32[128]", arg175_1: "f32[128]", arg176_1: "f32[128]", arg177_1: "f32[512, 128]", arg178_1: "f32[512]", arg179_1: "f32[128, 512]", arg180_1: "f32[128]", arg181_1: "f32[128]", arg182_1: "f32[128]", arg183_1: "f32[512, 128]", arg184_1: "f32[512]", arg185_1: "f32[128, 512]", arg186_1: "f32[128]", arg187_1: "f32[128]", arg188_1: "f32[128]", arg189_1: "f32[512, 128]", arg190_1: "f32[512]", arg191_1: "f32[512]", arg192_1: "f32[512]", arg193_1: "f32[128, 512]", arg194_1: "f32[128]", arg195_1: "f32[128]", arg196_1: "f32[128]", arg197_1: "f32[128, 512]", arg198_1: "f32[128]", arg199_1: "f32[128]", arg200_1: "f32[128]", arg201_1: "f32[128, 128]", arg202_1: "f32[128]", arg203_1: "f32[128, 128]", arg204_1: "f32[128]", arg205_1: "f32[128, 512]", arg206_1: "f32[128]", arg207_1: "f32[128, 128]", arg208_1: "f32[128]", arg209_1: "f32[128]", arg210_1: "f32[128]", arg211_1: "f32[512, 128]", arg212_1: "f32[512]", arg213_1: "f32[128, 512]", arg214_1: "f32[128]", arg215_1: "f32[128]", arg216_1: "f32[128]", arg217_1: "f32[512, 128]", arg218_1: "f32[512]", arg219_1: "f32[128, 512]", arg220_1: "f32[128]", arg221_1: "f32[128]", arg222_1: "f32[128]", arg223_1: "f32[512, 128]", arg224_1: "f32[512]", arg225_1: "f32[128, 512]", arg226_1: "f32[128]", arg227_1: "f32[128]", arg228_1: "f32[128]", arg229_1: "f32[512, 128]", arg230_1: "f32[512]", arg231_1: "f32[128, 512]", arg232_1: "f32[128]", arg233_1: "f32[128]", arg234_1: "f32[128]", arg235_1: "f32[512, 128]", arg236_1: "f32[512]", arg237_1: "f32[512]", arg238_1: "f32[512]", arg239_1: "f32[128, 512]", arg240_1: "f32[128]", arg241_1: "f32[128]", arg242_1: "f32[128]", arg243_1: "f32[128, 512]", arg244_1: "f32[128]", arg245_1: "f32[128]", arg246_1: "f32[128]", arg247_1: "f32[128, 128]", arg248_1: "f32[128]", arg249_1: "f32[128, 128]", arg250_1: "f32[128]", arg251_1: "f32[128, 512]", arg252_1: "f32[128]", arg253_1: "f32[128, 128]", arg254_1: "f32[128]", arg255_1: "f32[128]", arg256_1: "f32[128]", arg257_1: "f32[512, 128]", arg258_1: "f32[512]", arg259_1: "f32[128, 512]", arg260_1: "f32[128]", arg261_1: "f32[128]", arg262_1: "f32[128]", arg263_1: "f32[512, 128]", arg264_1: "f32[512]", arg265_1: "f32[128, 512]", arg266_1: "f32[128]", arg267_1: "f32[128]", arg268_1: "f32[128]", arg269_1: "f32[512, 128]", arg270_1: "f32[512]", arg271_1: "f32[128, 512]", arg272_1: "f32[128]", arg273_1: "f32[128]", arg274_1: "f32[128]", arg275_1: "f32[512, 128]", arg276_1: "f32[512]", arg277_1: "f32[128, 512]", arg278_1: "f32[128]", arg279_1: "f32[128]", arg280_1: "f32[128]", arg281_1: "f32[512, 128]", arg282_1: "f32[512]", arg283_1: "f32[512]", arg284_1: "f32[512]", arg285_1: "f32[128, 512]", arg286_1: "f32[128]", arg287_1: "f32[128]", arg288_1: "f32[128]", arg289_1: "f32[128, 512]", arg290_1: "f32[128]", arg291_1: "f32[128]", arg292_1: "f32[128]", arg293_1: "f32[128, 128]", arg294_1: "f32[128]", arg295_1: "f32[128, 128]", arg296_1: "f32[128]", arg297_1: "f32[128, 512]", arg298_1: "f32[128]", arg299_1: "f32[128, 128]", arg300_1: "f32[128]", arg301_1: "f32[128]", arg302_1: "f32[128]", arg303_1: "f32[512, 128]", arg304_1: "f32[512]", arg305_1: "f32[128, 512]", arg306_1: "f32[128]", arg307_1: "f32[128]", arg308_1: "f32[128]", arg309_1: "f32[512, 128]", arg310_1: "f32[512]", arg311_1: "f32[128, 512]", arg312_1: "f32[128]", arg313_1: "f32[128]", arg314_1: "f32[128]", arg315_1: "f32[512, 128]", arg316_1: "f32[512]", arg317_1: "f32[128, 512]", arg318_1: "f32[128]", arg319_1: "f32[128]", arg320_1: "f32[128]", arg321_1: "f32[512, 128]", arg322_1: "f32[512]", arg323_1: "f32[128, 512]", arg324_1: "f32[128]", arg325_1: "f32[128]", arg326_1: "f32[128]", arg327_1: "f32[512, 128]", arg328_1: "f32[512]", arg329_1: "f32[512]", arg330_1: "f32[512]", arg331_1: "f32[128, 512]", arg332_1: "f32[128]", arg333_1: "f32[128]", arg334_1: "f32[128]", arg335_1: "f32[128, 512]", arg336_1: "f32[128]", arg337_1: "f32[128]", arg338_1: "f32[128]", arg339_1: "f32[128, 128]", arg340_1: "f32[128]", arg341_1: "f32[128, 128]", arg342_1: "f32[128]", arg343_1: "f32[128, 512]", arg344_1: "f32[128]", arg345_1: "f32[128, 128]", arg346_1: "f32[128]", arg347_1: "f32[128]", arg348_1: "f32[128]", arg349_1: "f32[512, 128]", arg350_1: "f32[512]", arg351_1: "f32[128, 512]", arg352_1: "f32[128]", arg353_1: "f32[128]", arg354_1: "f32[128]", arg355_1: "f32[512, 128]", arg356_1: "f32[512]", arg357_1: "f32[128, 512]", arg358_1: "f32[128]", arg359_1: "f32[128]", arg360_1: "f32[128]", arg361_1: "f32[512, 128]", arg362_1: "f32[512]", arg363_1: "f32[128, 512]", arg364_1: "f32[128]", arg365_1: "f32[128]", arg366_1: "f32[128]", arg367_1: "f32[512, 128]", arg368_1: "f32[512]", arg369_1: "f32[128, 512]", arg370_1: "f32[128]", arg371_1: "f32[128]", arg372_1: "f32[128]", arg373_1: "f32[512, 128]", arg374_1: "f32[512]", arg375_1: "f32[512]", arg376_1: "f32[512]", arg377_1: "f32[128, 512]", arg378_1: "f32[128]", arg379_1: "f32[128]", arg380_1: "f32[128]", arg381_1: "f32[128, 512]", arg382_1: "f32[128]", arg383_1: "f32[128]", arg384_1: "f32[128]", arg385_1: "f32[128, 128]", arg386_1: "f32[128]", arg387_1: "f32[128, 128]", arg388_1: "f32[128]", arg389_1: "f32[128, 512]", arg390_1: "f32[128]", arg391_1: "f32[128, 128]", arg392_1: "f32[128]", arg393_1: "f32[128]", arg394_1: "f32[128]", arg395_1: "f32[512, 128]", arg396_1: "f32[512]", arg397_1: "f32[128, 512]", arg398_1: "f32[128]", arg399_1: "f32[128]", arg400_1: "f32[128]", arg401_1: "f32[512, 128]", arg402_1: "f32[512]", arg403_1: "f32[128, 512]", arg404_1: "f32[128]", arg405_1: "f32[128]", arg406_1: "f32[128]", arg407_1: "f32[512, 128]", arg408_1: "f32[512]", arg409_1: "f32[128, 512]", arg410_1: "f32[128]", arg411_1: "f32[128]", arg412_1: "f32[128]", arg413_1: "f32[512, 128]", arg414_1: "f32[512]", arg415_1: "f32[128, 512]", arg416_1: "f32[128]", arg417_1: "f32[128]", arg418_1: "f32[128]", arg419_1: "f32[512, 128]", arg420_1: "f32[512]", arg421_1: "f32[512]", arg422_1: "f32[512]", arg423_1: "f32[128, 512]", arg424_1: "f32[128]", arg425_1: "f32[128]", arg426_1: "f32[128]", arg427_1: "f32[128, 512]", arg428_1: "f32[128]", arg429_1: "f32[128]", arg430_1: "f32[128]", arg431_1: "f32[128, 128]", arg432_1: "f32[128]", arg433_1: "f32[128, 128]", arg434_1: "f32[128]", arg435_1: "f32[128, 512]", arg436_1: "f32[128]", arg437_1: "f32[128, 128]", arg438_1: "f32[128]", arg439_1: "f32[128]", arg440_1: "f32[128]", arg441_1: "f32[512, 128]", arg442_1: "f32[512]", arg443_1: "f32[128, 512]", arg444_1: "f32[128]", arg445_1: "f32[128]", arg446_1: "f32[128]", arg447_1: "f32[512, 128]", arg448_1: "f32[512]", arg449_1: "f32[128, 512]", arg450_1: "f32[128]", arg451_1: "f32[128]", arg452_1: "f32[128]", arg453_1: "f32[512, 128]", arg454_1: "f32[512]", arg455_1: "f32[128, 512]", arg456_1: "f32[128]", arg457_1: "f32[128]", arg458_1: "f32[128]", arg459_1: "f32[512, 128]", arg460_1: "f32[512]", arg461_1: "f32[128, 512]", arg462_1: "f32[128]", arg463_1: "f32[128]", arg464_1: "f32[128]", arg465_1: "f32[512, 128]", arg466_1: "f32[512]", arg467_1: "f32[512]", arg468_1: "f32[512]", arg469_1: "f32[128, 512]", arg470_1: "f32[128]", arg471_1: "f32[128]", arg472_1: "f32[128]", arg473_1: "f32[128, 512]", arg474_1: "f32[128]", arg475_1: "f32[128]", arg476_1: "f32[128]", arg477_1: "f32[128, 128]", arg478_1: "f32[128]", arg479_1: "f32[128, 128]", arg480_1: "f32[128]", arg481_1: "f32[128, 512]", arg482_1: "f32[128]", arg483_1: "f32[128, 128]", arg484_1: "f32[128]", arg485_1: "f32[128]", arg486_1: "f32[128]", arg487_1: "f32[512, 128]", arg488_1: "f32[512]", arg489_1: "f32[128, 512]", arg490_1: "f32[128]", arg491_1: "f32[128]", arg492_1: "f32[128]", arg493_1: "f32[512, 128]", arg494_1: "f32[512]", arg495_1: "f32[128, 512]", arg496_1: "f32[128]", arg497_1: "f32[128]", arg498_1: "f32[128]", arg499_1: "f32[512, 128]", arg500_1: "f32[512]", arg501_1: "f32[128, 512]", arg502_1: "f32[128]", arg503_1: "f32[128]", arg504_1: "f32[128]", arg505_1: "f32[512, 128]", arg506_1: "f32[512]", arg507_1: "f32[128, 512]", arg508_1: "f32[128]", arg509_1: "f32[128]", arg510_1: "f32[128]", arg511_1: "f32[512, 128]", arg512_1: "f32[512]", arg513_1: "f32[512]", arg514_1: "f32[512]", arg515_1: "f32[128, 512]", arg516_1: "f32[128]", arg517_1: "f32[128]", arg518_1: "f32[128]", arg519_1: "f32[128, 512]", arg520_1: "f32[128]", arg521_1: "f32[128]", arg522_1: "f32[128]", arg523_1: "f32[128, 128]", arg524_1: "f32[128]", arg525_1: "f32[128, 128]", arg526_1: "f32[128]", arg527_1: "f32[128, 512]", arg528_1: "f32[128]", arg529_1: "f32[128, 128]", arg530_1: "f32[128]", arg531_1: "f32[128]", arg532_1: "f32[128]", arg533_1: "f32[512, 128]", arg534_1: "f32[512]", arg535_1: "f32[128, 512]", arg536_1: "f32[128]", arg537_1: "f32[128]", arg538_1: "f32[128]", arg539_1: "f32[512, 128]", arg540_1: "f32[512]", arg541_1: "f32[128, 512]", arg542_1: "f32[128]", arg543_1: "f32[128]", arg544_1: "f32[128]", arg545_1: "f32[512, 128]", arg546_1: "f32[512]", arg547_1: "f32[128, 512]", arg548_1: "f32[128]", arg549_1: "f32[128]", arg550_1: "f32[128]", arg551_1: "f32[512, 128]", arg552_1: "f32[512]", arg553_1: "f32[128, 512]", arg554_1: "f32[128]", arg555_1: "f32[128]", arg556_1: "f32[128]", arg557_1: "f32[512, 128]", arg558_1: "f32[512]", arg559_1: "f32[512]", arg560_1: "f32[512]", arg561_1: "f32[128, 512]", arg562_1: "f32[128]", arg563_1: "f32[128]", arg564_1: "f32[128]", arg565_1: "f32[128, 512]", arg566_1: "f32[128]", arg567_1: "f32[128]", arg568_1: "f32[128]", arg569_1: "f32[128, 128]", arg570_1: "f32[128]", arg571_1: "f32[128, 128]", arg572_1: "f32[128]", arg573_1: "f32[128, 512]", arg574_1: "f32[128]", arg575_1: "f32[128, 128]", arg576_1: "f32[128]", arg577_1: "f32[128]", arg578_1: "f32[128]", arg579_1: "f32[512, 128]", arg580_1: "f32[512]", arg581_1: "f32[128, 512]", arg582_1: "f32[128]", arg583_1: "f32[128]", arg584_1: "f32[128]", arg585_1: "f32[512, 128]", arg586_1: "f32[512]", arg587_1: "f32[128, 512]", arg588_1: "f32[128]", arg589_1: "f32[128]", arg590_1: "f32[128]", arg591_1: "f32[512, 128]", arg592_1: "f32[512]", arg593_1: "f32[128, 512]", arg594_1: "f32[128]", arg595_1: "f32[128]", arg596_1: "f32[128]", arg597_1: "f32[512, 128]", arg598_1: "f32[512]", arg599_1: "f32[128, 512]", arg600_1: "f32[128]", arg601_1: "f32[128]", arg602_1: "f32[128]", arg603_1: "f32[512, 128]", arg604_1: "f32[512]", arg605_1: "f32[512]", arg606_1: "f32[512]", arg607_1: "f32[128, 512]", arg608_1: "f32[128]", arg609_1: "f32[128]", arg610_1: "f32[128]", arg611_1: "f32[128, 512]", arg612_1: "f32[128]", arg613_1: "f32[128]", arg614_1: "f32[128]", arg615_1: "f32[128, 128]", arg616_1: "f32[128]", arg617_1: "f32[128, 128]", arg618_1: "f32[128]", arg619_1: "f32[128, 512]", arg620_1: "f32[128]", arg621_1: "f32[128, 128]", arg622_1: "f32[128]", arg623_1: "f32[128]", arg624_1: "f32[128]", arg625_1: "f32[512, 128]", arg626_1: "f32[512]", arg627_1: "f32[128, 512]", arg628_1: "f32[128]", arg629_1: "f32[128]", arg630_1: "f32[128]", arg631_1: "f32[512, 128]", arg632_1: "f32[512]", arg633_1: "f32[128, 512]", arg634_1: "f32[128]", arg635_1: "f32[128]", arg636_1: "f32[128]", arg637_1: "f32[512, 128]", arg638_1: "f32[512]", arg639_1: "f32[128, 512]", arg640_1: "f32[128]", arg641_1: "f32[128]", arg642_1: "f32[128]", arg643_1: "f32[512, 128]", arg644_1: "f32[512]", arg645_1: "f32[128, 512]", arg646_1: "f32[128]", arg647_1: "f32[128]", arg648_1: "f32[128]", arg649_1: "f32[512, 128]", arg650_1: "f32[512]", arg651_1: "f32[512]", arg652_1: "f32[512]", arg653_1: "f32[128, 512]", arg654_1: "f32[128]", arg655_1: "f32[128]", arg656_1: "f32[128]", arg657_1: "f32[128, 512]", arg658_1: "f32[128]", arg659_1: "f32[128]", arg660_1: "f32[128]", arg661_1: "f32[128, 128]", arg662_1: "f32[128]", arg663_1: "f32[128, 128]", arg664_1: "f32[128]", arg665_1: "f32[128, 512]", arg666_1: "f32[128]", arg667_1: "f32[128, 128]", arg668_1: "f32[128]", arg669_1: "f32[128]", arg670_1: "f32[128]", arg671_1: "f32[512, 128]", arg672_1: "f32[512]", arg673_1: "f32[128, 512]", arg674_1: "f32[128]", arg675_1: "f32[128]", arg676_1: "f32[128]", arg677_1: "f32[512, 128]", arg678_1: "f32[512]", arg679_1: "f32[128, 512]", arg680_1: "f32[128]", arg681_1: "f32[128]", arg682_1: "f32[128]", arg683_1: "f32[512, 128]", arg684_1: "f32[512]", arg685_1: "f32[128, 512]", arg686_1: "f32[128]", arg687_1: "f32[128]", arg688_1: "f32[128]", arg689_1: "f32[512, 128]", arg690_1: "f32[512]", arg691_1: "f32[128, 512]", arg692_1: "f32[128]", arg693_1: "f32[128]", arg694_1: "f32[128]", arg695_1: "f32[512, 128]", arg696_1: "f32[512]", arg697_1: "f32[512]", arg698_1: "f32[512]", arg699_1: "f32[128, 512]", arg700_1: "f32[128]", arg701_1: "f32[128]", arg702_1: "f32[128]", arg703_1: "f32[128, 512]", arg704_1: "f32[128]", arg705_1: "f32[128]", arg706_1: "f32[128]", arg707_1: "f32[128, 128]", arg708_1: "f32[128]", arg709_1: "f32[128, 128]", arg710_1: "f32[128]", arg711_1: "f32[128, 512]", arg712_1: "f32[128]", arg713_1: "f32[128, 128]", arg714_1: "f32[128]", arg715_1: "f32[128]", arg716_1: "f32[128]", arg717_1: "f32[512, 128]", arg718_1: "f32[512]", arg719_1: "f32[128, 512]", arg720_1: "f32[128]", arg721_1: "f32[128]", arg722_1: "f32[128]", arg723_1: "f32[512, 128]", arg724_1: "f32[512]", arg725_1: "f32[128, 512]", arg726_1: "f32[128]", arg727_1: "f32[128]", arg728_1: "f32[128]", arg729_1: "f32[512, 128]", arg730_1: "f32[512]", arg731_1: "f32[128, 512]", arg732_1: "f32[128]", arg733_1: "f32[128]", arg734_1: "f32[128]", arg735_1: "f32[512, 128]", arg736_1: "f32[512]", arg737_1: "f32[128, 512]", arg738_1: "f32[128]", arg739_1: "f32[128]", arg740_1: "f32[128]", arg741_1: "f32[512, 128]", arg742_1: "f32[512]", arg743_1: "f32[512]", arg744_1: "f32[512]", arg745_1: "f32[128, 512]", arg746_1: "f32[128]", arg747_1: "f32[128]", arg748_1: "f32[128]", arg749_1: "f32[128, 512]", arg750_1: "f32[128]", arg751_1: "f32[128]", arg752_1: "f32[128]", arg753_1: "f32[128, 128]", arg754_1: "f32[128]", arg755_1: "f32[128, 128]", arg756_1: "f32[128]", arg757_1: "f32[128, 512]", arg758_1: "f32[128]", arg759_1: "f32[128, 128]", arg760_1: "f32[128]", arg761_1: "f32[128]", arg762_1: "f32[128]", arg763_1: "f32[512, 128]", arg764_1: "f32[512]", arg765_1: "f32[128, 512]", arg766_1: "f32[128]", arg767_1: "f32[128]", arg768_1: "f32[128]", arg769_1: "f32[512, 128]", arg770_1: "f32[512]", arg771_1: "f32[128, 512]", arg772_1: "f32[128]", arg773_1: "f32[128]", arg774_1: "f32[128]", arg775_1: "f32[512, 128]", arg776_1: "f32[512]", arg777_1: "f32[128, 512]", arg778_1: "f32[128]", arg779_1: "f32[128]", arg780_1: "f32[128]", arg781_1: "f32[512, 128]", arg782_1: "f32[512]", arg783_1: "f32[128, 512]", arg784_1: "f32[128]", arg785_1: "f32[128]", arg786_1: "f32[128]", arg787_1: "f32[512, 128]", arg788_1: "f32[512]", arg789_1: "f32[512]", arg790_1: "f32[512]", arg791_1: "f32[128, 512]", arg792_1: "f32[128]", arg793_1: "f32[128]", arg794_1: "f32[128]", arg795_1: "f32[128, 512]", arg796_1: "f32[128]", arg797_1: "f32[128]", arg798_1: "f32[128]", arg799_1: "f32[128, 128]", arg800_1: "f32[128]", arg801_1: "f32[128, 128]", arg802_1: "f32[128]", arg803_1: "f32[128, 512]", arg804_1: "f32[128]", arg805_1: "f32[128, 128]", arg806_1: "f32[128]", arg807_1: "f32[128]", arg808_1: "f32[128]", arg809_1: "f32[512, 128]", arg810_1: "f32[512]", arg811_1: "f32[128, 512]", arg812_1: "f32[128]", arg813_1: "f32[128]", arg814_1: "f32[128]", arg815_1: "f32[512, 128]", arg816_1: "f32[512]", arg817_1: "f32[128, 512]", arg818_1: "f32[128]", arg819_1: "f32[128]", arg820_1: "f32[128]", arg821_1: "f32[512, 128]", arg822_1: "f32[512]", arg823_1: "f32[128, 512]", arg824_1: "f32[128]", arg825_1: "f32[128]", arg826_1: "f32[128]", arg827_1: "f32[512, 128]", arg828_1: "f32[512]", arg829_1: "f32[128, 512]", arg830_1: "f32[128]", arg831_1: "f32[128]", arg832_1: "f32[128]", arg833_1: "f32[512, 128]", arg834_1: "f32[512]", arg835_1: "f32[512]", arg836_1: "f32[512]", arg837_1: "f32[128, 512]", arg838_1: "f32[128]", arg839_1: "f32[128]", arg840_1: "f32[128]", arg841_1: "f32[128, 512]", arg842_1: "f32[128]", arg843_1: "f32[128]", arg844_1: "f32[128]", arg845_1: "f32[128, 128]", arg846_1: "f32[128]", arg847_1: "f32[128, 128]", arg848_1: "f32[128]", arg849_1: "f32[128, 512]", arg850_1: "f32[128]", arg851_1: "f32[128, 128]", arg852_1: "f32[128]", arg853_1: "f32[128]", arg854_1: "f32[128]", arg855_1: "f32[512, 128]", arg856_1: "f32[512]", arg857_1: "f32[128, 512]", arg858_1: "f32[128]", arg859_1: "f32[128]", arg860_1: "f32[128]", arg861_1: "f32[512, 128]", arg862_1: "f32[512]", arg863_1: "f32[128, 512]", arg864_1: "f32[128]", arg865_1: "f32[128]", arg866_1: "f32[128]", arg867_1: "f32[512, 128]", arg868_1: "f32[512]", arg869_1: "f32[128, 512]", arg870_1: "f32[128]", arg871_1: "f32[128]", arg872_1: "f32[128]", arg873_1: "f32[512, 128]", arg874_1: "f32[512]", arg875_1: "f32[128, 512]", arg876_1: "f32[128]", arg877_1: "f32[128]", arg878_1: "f32[128]", arg879_1: "f32[512, 128]", arg880_1: "f32[512]", arg881_1: "f32[512]", arg882_1: "f32[512]", arg883_1: "f32[128, 512]", arg884_1: "f32[128]", arg885_1: "f32[128]", arg886_1: "f32[128]", arg887_1: "f32[128, 512]", arg888_1: "f32[128]", arg889_1: "f32[128]", arg890_1: "f32[128]", arg891_1: "f32[128, 128]", arg892_1: "f32[128]", arg893_1: "f32[128, 128]", arg894_1: "f32[128]", arg895_1: "f32[128, 512]", arg896_1: "f32[128]", arg897_1: "f32[128, 128]", arg898_1: "f32[128]", arg899_1: "f32[128]", arg900_1: "f32[128]", arg901_1: "f32[512, 128]", arg902_1: "f32[512]", arg903_1: "f32[128, 512]", arg904_1: "f32[128]", arg905_1: "f32[128]", arg906_1: "f32[128]", arg907_1: "f32[512, 128]", arg908_1: "f32[512]", arg909_1: "f32[128, 512]", arg910_1: "f32[128]", arg911_1: "f32[128]", arg912_1: "f32[128]", arg913_1: "f32[512, 128]", arg914_1: "f32[512]", arg915_1: "f32[128, 512]", arg916_1: "f32[128]", arg917_1: "f32[128]", arg918_1: "f32[128]", arg919_1: "f32[512, 128]", arg920_1: "f32[512]", arg921_1: "f32[128, 512]", arg922_1: "f32[128]", arg923_1: "f32[128]", arg924_1: "f32[128]", arg925_1: "f32[512, 128]", arg926_1: "f32[512]", arg927_1: "f32[512]", arg928_1: "f32[512]", arg929_1: "f32[128, 512]", arg930_1: "f32[128]", arg931_1: "f32[128]", arg932_1: "f32[128]", arg933_1: "f32[128, 512]", arg934_1: "f32[128]", arg935_1: "f32[128]", arg936_1: "f32[128]", arg937_1: "f32[128, 128]", arg938_1: "f32[128]", arg939_1: "f32[128, 128]", arg940_1: "f32[128]", arg941_1: "f32[128, 512]", arg942_1: "f32[128]", arg943_1: "f32[128, 128]", arg944_1: "f32[128]", arg945_1: "f32[128]", arg946_1: "f32[128]", arg947_1: "f32[512, 128]", arg948_1: "f32[512]", arg949_1: "f32[128, 512]", arg950_1: "f32[128]", arg951_1: "f32[128]", arg952_1: "f32[128]", arg953_1: "f32[512, 128]", arg954_1: "f32[512]", arg955_1: "f32[128, 512]", arg956_1: "f32[128]", arg957_1: "f32[128]", arg958_1: "f32[128]", arg959_1: "f32[512, 128]", arg960_1: "f32[512]", arg961_1: "f32[128, 512]", arg962_1: "f32[128]", arg963_1: "f32[128]", arg964_1: "f32[128]", arg965_1: "f32[512, 128]", arg966_1: "f32[512]", arg967_1: "f32[128, 512]", arg968_1: "f32[128]", arg969_1: "f32[128]", arg970_1: "f32[128]", arg971_1: "f32[512, 128]", arg972_1: "f32[512]", arg973_1: "f32[512]", arg974_1: "f32[512]", arg975_1: "f32[128, 512]", arg976_1: "f32[128]", arg977_1: "f32[128]", arg978_1: "f32[128]", arg979_1: "f32[128, 512]", arg980_1: "f32[128]", arg981_1: "f32[128]", arg982_1: "f32[128]", arg983_1: "f32[128, 128]", arg984_1: "f32[128]", arg985_1: "f32[128, 128]", arg986_1: "f32[128]", arg987_1: "f32[128, 512]", arg988_1: "f32[128]", arg989_1: "f32[128, 128]", arg990_1: "f32[128]", arg991_1: "f32[128]", arg992_1: "f32[128]", arg993_1: "f32[512, 128]", arg994_1: "f32[512]", arg995_1: "f32[128, 512]", arg996_1: "f32[128]", arg997_1: "f32[128]", arg998_1: "f32[128]", arg999_1: "f32[512, 128]", arg1000_1: "f32[512]", arg1001_1: "f32[128, 512]", arg1002_1: "f32[128]", arg1003_1: "f32[128]", arg1004_1: "f32[128]", arg1005_1: "f32[512, 128]", arg1006_1: "f32[512]", arg1007_1: "f32[128, 512]", arg1008_1: "f32[128]", arg1009_1: "f32[128]", arg1010_1: "f32[128]", arg1011_1: "f32[512, 128]", arg1012_1: "f32[512]", arg1013_1: "f32[128, 512]", arg1014_1: "f32[128]", arg1015_1: "f32[128]", arg1016_1: "f32[128]", arg1017_1: "f32[512, 128]", arg1018_1: "f32[512]", arg1019_1: "f32[512]", arg1020_1: "f32[512]", arg1021_1: "f32[128, 512]", arg1022_1: "f32[128]", arg1023_1: "f32[128]", arg1024_1: "f32[128]", arg1025_1: "f32[128, 512]", arg1026_1: "f32[128]", arg1027_1: "f32[128]", arg1028_1: "f32[128]", arg1029_1: "f32[128, 128]", arg1030_1: "f32[128]", arg1031_1: "f32[128, 128]", arg1032_1: "f32[128]", arg1033_1: "f32[128, 512]", arg1034_1: "f32[128]", arg1035_1: "f32[128, 128]", arg1036_1: "f32[128]", arg1037_1: "f32[128]", arg1038_1: "f32[128]", arg1039_1: "f32[512, 128]", arg1040_1: "f32[512]", arg1041_1: "f32[128, 512]", arg1042_1: "f32[128]", arg1043_1: "f32[128]", arg1044_1: "f32[128]", arg1045_1: "f32[512, 128]", arg1046_1: "f32[512]", arg1047_1: "f32[128, 512]", arg1048_1: "f32[128]", arg1049_1: "f32[128]", arg1050_1: "f32[128]", arg1051_1: "f32[512, 128]", arg1052_1: "f32[512]", arg1053_1: "f32[128, 512]", arg1054_1: "f32[128]", arg1055_1: "f32[128]", arg1056_1: "f32[128]", arg1057_1: "f32[512, 128]", arg1058_1: "f32[512]", arg1059_1: "f32[128, 512]", arg1060_1: "f32[128]", arg1061_1: "f32[128]", arg1062_1: "f32[128]", arg1063_1: "f32[512, 128]", arg1064_1: "f32[512]", arg1065_1: "f32[512]", arg1066_1: "f32[512]", arg1067_1: "f32[128, 512]", arg1068_1: "f32[128]", arg1069_1: "f32[128]", arg1070_1: "f32[128]", arg1071_1: "f32[128, 512]", arg1072_1: "f32[128]", arg1073_1: "f32[128]", arg1074_1: "f32[128]", arg1075_1: "f32[128, 128]", arg1076_1: "f32[128]", arg1077_1: "f32[128, 128]", arg1078_1: "f32[128]", arg1079_1: "f32[128, 512]", arg1080_1: "f32[128]", arg1081_1: "f32[128, 128]", arg1082_1: "f32[128]", arg1083_1: "f32[128]", arg1084_1: "f32[128]", arg1085_1: "f32[512, 128]", arg1086_1: "f32[512]", arg1087_1: "f32[128, 512]", arg1088_1: "f32[128]", arg1089_1: "f32[128]", arg1090_1: "f32[128]", arg1091_1: "f32[512, 128]", arg1092_1: "f32[512]", arg1093_1: "f32[128, 512]", arg1094_1: "f32[128]", arg1095_1: "f32[128]", arg1096_1: "f32[128]", arg1097_1: "f32[512, 128]", arg1098_1: "f32[512]", arg1099_1: "f32[128, 512]", arg1100_1: "f32[128]", arg1101_1: "f32[128]", arg1102_1: "f32[128]", arg1103_1: "f32[512, 128]", arg1104_1: "f32[512]", arg1105_1: "f32[128, 512]", arg1106_1: "f32[128]", arg1107_1: "f32[128]", arg1108_1: "f32[128]", arg1109_1: "f32[512, 128]", arg1110_1: "f32[512]", arg1111_1: "f32[512]", arg1112_1: "f32[512]", arg1113_1: "f32[512, 512]", arg1114_1: "f32[512]", arg1115_1: "f32[512]", arg1116_1: "f32[512]", arg1117_1: "f32[384, 30522]", arg1118_1: "f32[30522]", arg1119_1: "i64[256, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:113 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[256, 128, 128]" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:125 in forward, code: nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
        slice_2: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 1, 9223372036854775807)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_2, [0, 0, 0, 1, 0, 0], 0.0);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:127 in forward, code: nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
        slice_3: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 0, -1)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_3, [0, 0, 1, 0, 0, 0], 0.0);  slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:123 in forward, code: inputs_embeds = torch.cat(
        cat: "f32[256, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:132 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        view: "f32[32768, 384]" = torch.ops.aten.reshape.default(cat, [32768, 384]);  cat = None
        permute: "f32[384, 512]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg4_1, view, permute);  arg4_1 = view = permute = None
        view_1: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm, [256, 128, 512]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:108 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 128);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:136 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_1: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg5_1, slice_1);  arg5_1 = slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:111 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        full_default: "i64[256, 128]" = torch.ops.aten.full.default([256, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:137 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_2: "f32[256, 128, 512]" = torch.ops.aten.embedding.default(arg6_1, full_default);  arg6_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg7_1);  add_1 = arg7_1 = None
        add_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul, arg8_1);  mul = arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_4: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_2, [32768, 512])
        permute_2: "f32[512, 128]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg14_1, view_4, permute_2);  arg14_1 = view_4 = permute_2 = None
        view_5: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_2, [256, 128, 128]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_2: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_5, arg15_1);  view_5 = arg15_1 = None
        add_6: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_2, arg16_1);  mul_2 = arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_6: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_6, [32768, 128])
        permute_3: "f32[128, 128]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_3: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg18_1, view_6, permute_3);  arg18_1 = view_6 = permute_3 = None
        view_7: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_3, [256, 128, 128]);  addmm_3 = None
        view_8: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_7, [256, 128, -1, 32]);  view_7 = None
        permute_4: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_3: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_4, 0.4204482076268573);  permute_4 = None
        expand_1: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_3, [256, 4, 128, 32]);  mul_3 = None
        clone_1: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_15: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_1, [1024, 128, 32]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_9: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_6, [32768, 128]);  add_6 = None
        permute_5: "f32[128, 128]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg20_1, view_9, permute_5);  arg20_1 = view_9 = permute_5 = None
        view_10: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_4, [256, 128, 128]);  addmm_4 = None
        view_11: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_10, [256, 128, -1, 32]);  view_10 = None
        permute_6: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_9: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2]);  permute_6 = None
        mul_4: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_9, 0.4204482076268573);  permute_9 = None
        expand_2: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_4, [256, 4, 32, 128]);  mul_4 = None
        clone_2: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_16: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_2, [1024, 32, 128]);  clone_2 = None
        bmm: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm, [256, 4, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[128]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[256, 1, 128, 128]" = torch.ops.aten.expand.default(ge, [256, -1, 128, 128]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        add_7: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_17, where);  view_17 = where = None
        eq: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_7, -inf)
        logical_not: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_3: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_7, [-1], True)
        sub: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_7, amax);  add_7 = amax = None
        exp: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_1: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_default_3, div);  logical_not_1 = full_default_3 = div = None
        expand_3: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_1, [256, 4, 128, 128]);  where_1 = None
        view_18: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [1024, 128, 128]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_12: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_2, [32768, 512])
        permute_7: "f32[512, 128]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_5: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg22_1, view_12, permute_7);  arg22_1 = view_12 = permute_7 = None
        view_13: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_5, [256, 128, 128]);  addmm_5 = None
        view_14: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_13, [256, 128, -1, 32]);  view_13 = None
        permute_8: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_8, [256, 4, 128, 32]);  permute_8 = None
        clone_3: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_19: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_3, [1024, 128, 32]);  clone_3 = None
        bmm_1: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_18, view_19);  view_18 = view_19 = None
        view_20: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_1, [256, 4, 128, 32]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_10: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        clone_4: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_21: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_4, [256, 128, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_22: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_21, [32768, 128]);  view_21 = None
        permute_11: "f32[128, 128]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_6: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg24_1, view_22, permute_11);  arg24_1 = view_22 = permute_11 = None
        view_23: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_6, [256, 128, 128]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_2: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_2, [32768, 512])
        permute_1: "f32[512, 128]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg10_1, view_2, permute_1);  arg10_1 = view_2 = permute_1 = None
        view_3: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_1, [256, 128, 128]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_3, arg11_1);  view_3 = arg11_1 = None
        add_5: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_1, arg12_1);  mul_1 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_8: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_23, add_5);  view_23 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_5: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_8, arg25_1);  add_8 = arg25_1 = None
        add_9: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_5, arg26_1);  mul_5 = arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_24: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_9, [32768, 128])
        permute_12: "f32[128, 512]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_7: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg28_1, view_24, permute_12);  arg28_1 = view_24 = permute_12 = None
        view_25: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_7, [256, 128, 512]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_25);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_26: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu, [32768, 512]);  relu = None
        permute_13: "f32[512, 128]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_8: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg30_1, view_26, permute_13);  arg30_1 = view_26 = permute_13 = None
        view_27: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_8, [256, 128, 128]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_10: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_27, add_9);  view_27 = add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_6: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_10, arg31_1);  add_10 = arg31_1 = None
        add_11: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, arg32_1);  mul_6 = arg32_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_28: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_11, [32768, 128])
        permute_14: "f32[128, 512]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_9: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg34_1, view_28, permute_14);  arg34_1 = view_28 = permute_14 = None
        view_29: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_9, [256, 128, 512]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_1: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_29);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_30: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_1, [32768, 512]);  relu_1 = None
        permute_15: "f32[512, 128]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_10: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg36_1, view_30, permute_15);  arg36_1 = view_30 = permute_15 = None
        view_31: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_10, [256, 128, 128]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_12: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_31, add_11);  view_31 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_7: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_12, arg37_1);  add_12 = arg37_1 = None
        add_13: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_7, arg38_1);  mul_7 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_32: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_13, [32768, 128])
        permute_16: "f32[128, 512]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_11: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg40_1, view_32, permute_16);  arg40_1 = view_32 = permute_16 = None
        view_33: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_11, [256, 128, 512]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_2: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_33);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_34: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_2, [32768, 512]);  relu_2 = None
        permute_17: "f32[512, 128]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_12: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg42_1, view_34, permute_17);  arg42_1 = view_34 = permute_17 = None
        view_35: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_12, [256, 128, 128]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_14: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_35, add_13);  view_35 = add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_8: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, arg43_1);  add_14 = arg43_1 = None
        add_15: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_8, arg44_1);  mul_8 = arg44_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_36: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_15, [32768, 128])
        permute_18: "f32[128, 512]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_13: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg46_1, view_36, permute_18);  arg46_1 = view_36 = permute_18 = None
        view_37: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_13, [256, 128, 512]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_3: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_37);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_38: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_3, [32768, 512]);  relu_3 = None
        permute_19: "f32[512, 128]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_14: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg48_1, view_38, permute_19);  arg48_1 = view_38 = permute_19 = None
        view_39: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_14, [256, 128, 128]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_16: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_39, add_15);  view_39 = add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_9: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_16, arg49_1);  add_16 = arg49_1 = None
        add_17: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_9, arg50_1);  mul_9 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_40: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_17, [32768, 128]);  add_17 = None
        permute_20: "f32[128, 512]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_15: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg52_1, view_40, permute_20);  arg52_1 = view_40 = permute_20 = None
        view_41: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_15, [256, 128, 512]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_18: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_41, add_2);  view_41 = add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_10: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_18, arg53_1);  add_18 = arg53_1 = None
        add_19: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_10, arg54_1);  mul_10 = arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_44: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_19, [32768, 512])
        permute_22: "f32[512, 128]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_17: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg60_1, view_44, permute_22);  arg60_1 = view_44 = permute_22 = None
        view_45: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_17, [256, 128, 128]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_12: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_45, arg61_1);  view_45 = arg61_1 = None
        add_21: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_12, arg62_1);  mul_12 = arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_46: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_21, [32768, 128])
        permute_23: "f32[128, 128]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_18: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg64_1, view_46, permute_23);  arg64_1 = view_46 = permute_23 = None
        view_47: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_18, [256, 128, 128]);  addmm_18 = None
        view_48: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_47, [256, 128, -1, 32]);  view_47 = None
        permute_24: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_13: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_24, 0.4204482076268573);  permute_24 = None
        expand_5: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_13, [256, 4, 128, 32]);  mul_13 = None
        clone_6: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_55: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_6, [1024, 128, 32]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_49: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_21, [32768, 128]);  add_21 = None
        permute_25: "f32[128, 128]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_19: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg66_1, view_49, permute_25);  arg66_1 = view_49 = permute_25 = None
        view_50: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_19, [256, 128, 128]);  addmm_19 = None
        view_51: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_50, [256, 128, -1, 32]);  view_50 = None
        permute_26: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_29: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        mul_14: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_29, 0.4204482076268573);  permute_29 = None
        expand_6: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_14, [256, 4, 32, 128]);  mul_14 = None
        clone_7: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_56: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_7, [1024, 32, 128]);  clone_7 = None
        bmm_2: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
        view_57: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [256, 4, 128, 128]);  bmm_2 = None
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_5, full_default_4);  full_default_5 = full_default_4 = None
        add_22: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_57, where_2);  view_57 = where_2 = None
        eq_1: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_22, -inf)
        logical_not_2: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_6: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_1: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
        sub_1: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_22, amax_1);  add_22 = amax_1 = None
        exp_1: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_2: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        where_3: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_3, full_default_6, div_1);  logical_not_3 = full_default_6 = div_1 = None
        expand_7: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_3, [256, 4, 128, 128]);  where_3 = None
        view_58: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_7, [1024, 128, 128]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_52: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_19, [32768, 512])
        permute_27: "f32[512, 128]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_20: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg68_1, view_52, permute_27);  arg68_1 = view_52 = permute_27 = None
        view_53: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_20, [256, 128, 128]);  addmm_20 = None
        view_54: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_53, [256, 128, -1, 32]);  view_53 = None
        permute_28: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_28, [256, 4, 128, 32]);  permute_28 = None
        clone_8: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_59: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_8, [1024, 128, 32]);  clone_8 = None
        bmm_3: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_3, [256, 4, 128, 32]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_9: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_9, [256, 128, -1]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_62: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_61, [32768, 128]);  view_61 = None
        permute_31: "f32[128, 128]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_21: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg70_1, view_62, permute_31);  arg70_1 = view_62 = permute_31 = None
        view_63: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_21, [256, 128, 128]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_42: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_19, [32768, 512])
        permute_21: "f32[512, 128]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_16: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg56_1, view_42, permute_21);  arg56_1 = view_42 = permute_21 = None
        view_43: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_16, [256, 128, 128]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_11: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_43, arg57_1);  view_43 = arg57_1 = None
        add_20: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_11, arg58_1);  mul_11 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_23: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_63, add_20);  view_63 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_15: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_23, arg71_1);  add_23 = arg71_1 = None
        add_24: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_15, arg72_1);  mul_15 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_24, [32768, 128])
        permute_32: "f32[128, 512]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_22: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg74_1, view_64, permute_32);  arg74_1 = view_64 = permute_32 = None
        view_65: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_22, [256, 128, 512]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_4: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_65);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_66: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_4, [32768, 512]);  relu_4 = None
        permute_33: "f32[512, 128]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_23: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg76_1, view_66, permute_33);  arg76_1 = view_66 = permute_33 = None
        view_67: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_23, [256, 128, 128]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_25: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_67, add_24);  view_67 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_16: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_25, arg77_1);  add_25 = arg77_1 = None
        add_26: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_16, arg78_1);  mul_16 = arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_68: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_26, [32768, 128])
        permute_34: "f32[128, 512]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_24: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg80_1, view_68, permute_34);  arg80_1 = view_68 = permute_34 = None
        view_69: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_24, [256, 128, 512]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_5: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_69);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_70: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_5, [32768, 512]);  relu_5 = None
        permute_35: "f32[512, 128]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_25: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg82_1, view_70, permute_35);  arg82_1 = view_70 = permute_35 = None
        view_71: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_25, [256, 128, 128]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_27: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_71, add_26);  view_71 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_17: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_27, arg83_1);  add_27 = arg83_1 = None
        add_28: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_17, arg84_1);  mul_17 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_72: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_28, [32768, 128])
        permute_36: "f32[128, 512]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_26: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg86_1, view_72, permute_36);  arg86_1 = view_72 = permute_36 = None
        view_73: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_26, [256, 128, 512]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_6: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_73);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_74: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_6, [32768, 512]);  relu_6 = None
        permute_37: "f32[512, 128]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_27: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg88_1, view_74, permute_37);  arg88_1 = view_74 = permute_37 = None
        view_75: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_27, [256, 128, 128]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_29: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_75, add_28);  view_75 = add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_18: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_29, arg89_1);  add_29 = arg89_1 = None
        add_30: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_18, arg90_1);  mul_18 = arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_76: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_30, [32768, 128])
        permute_38: "f32[128, 512]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_28: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg92_1, view_76, permute_38);  arg92_1 = view_76 = permute_38 = None
        view_77: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_28, [256, 128, 512]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_7: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_77);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_78: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_7, [32768, 512]);  relu_7 = None
        permute_39: "f32[512, 128]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_29: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg94_1, view_78, permute_39);  arg94_1 = view_78 = permute_39 = None
        view_79: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_29, [256, 128, 128]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_31: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_79, add_30);  view_79 = add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_19: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_31, arg95_1);  add_31 = arg95_1 = None
        add_32: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_19, arg96_1);  mul_19 = arg96_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_80: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_32, [32768, 128]);  add_32 = None
        permute_40: "f32[128, 512]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_30: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg98_1, view_80, permute_40);  arg98_1 = view_80 = permute_40 = None
        view_81: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_30, [256, 128, 512]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_33: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_81, add_19);  view_81 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_20: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_33, arg99_1);  add_33 = arg99_1 = None
        add_34: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_20, arg100_1);  mul_20 = arg100_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_84: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_34, [32768, 512])
        permute_42: "f32[512, 128]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_32: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg106_1, view_84, permute_42);  arg106_1 = view_84 = permute_42 = None
        view_85: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_32, [256, 128, 128]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_22: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_85, arg107_1);  view_85 = arg107_1 = None
        add_36: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, arg108_1);  mul_22 = arg108_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_86: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_36, [32768, 128])
        permute_43: "f32[128, 128]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_33: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg110_1, view_86, permute_43);  arg110_1 = view_86 = permute_43 = None
        view_87: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_33, [256, 128, 128]);  addmm_33 = None
        view_88: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_87, [256, 128, -1, 32]);  view_87 = None
        permute_44: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_23: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_44, 0.4204482076268573);  permute_44 = None
        expand_9: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_23, [256, 4, 128, 32]);  mul_23 = None
        clone_11: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_95: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_11, [1024, 128, 32]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_89: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_36, [32768, 128]);  add_36 = None
        permute_45: "f32[128, 128]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_34: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg112_1, view_89, permute_45);  arg112_1 = view_89 = permute_45 = None
        view_90: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_34, [256, 128, 128]);  addmm_34 = None
        view_91: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_90, [256, 128, -1, 32]);  view_90 = None
        permute_46: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_91, [0, 2, 1, 3]);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_49: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
        mul_24: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_49, 0.4204482076268573);  permute_49 = None
        expand_10: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_24, [256, 4, 32, 128]);  mul_24 = None
        clone_12: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_96: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_12, [1024, 32, 128]);  clone_12 = None
        bmm_4: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_95, view_96);  view_95 = view_96 = None
        view_97: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [256, 4, 128, 128]);  bmm_4 = None
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        add_37: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_97, where_4);  view_97 = where_4 = None
        eq_2: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_37, -inf)
        logical_not_4: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_9: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_2: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_37, [-1], True)
        sub_2: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_37, amax_2);  add_37 = amax_2 = None
        exp_2: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_3: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        where_5: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_5, full_default_9, div_2);  logical_not_5 = full_default_9 = div_2 = None
        expand_11: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_5, [256, 4, 128, 128]);  where_5 = None
        view_98: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_11, [1024, 128, 128]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_92: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_34, [32768, 512])
        permute_47: "f32[512, 128]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_35: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg114_1, view_92, permute_47);  arg114_1 = view_92 = permute_47 = None
        view_93: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_35, [256, 128, 128]);  addmm_35 = None
        view_94: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_93, [256, 128, -1, 32]);  view_93 = None
        permute_48: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_12: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_48, [256, 4, 128, 32]);  permute_48 = None
        clone_13: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_99: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_13, [1024, 128, 32]);  clone_13 = None
        bmm_5: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_98, view_99);  view_98 = view_99 = None
        view_100: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_5, [256, 4, 128, 32]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_50: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        clone_14: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_101: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_14, [256, 128, -1]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_102: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_101, [32768, 128]);  view_101 = None
        permute_51: "f32[128, 128]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_36: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg116_1, view_102, permute_51);  arg116_1 = view_102 = permute_51 = None
        view_103: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_36, [256, 128, 128]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_82: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_34, [32768, 512])
        permute_41: "f32[512, 128]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_31: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg102_1, view_82, permute_41);  arg102_1 = view_82 = permute_41 = None
        view_83: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_31, [256, 128, 128]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_21: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_83, arg103_1);  view_83 = arg103_1 = None
        add_35: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_21, arg104_1);  mul_21 = arg104_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_38: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_103, add_35);  view_103 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_25: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_38, arg117_1);  add_38 = arg117_1 = None
        add_39: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_25, arg118_1);  mul_25 = arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_39, [32768, 128])
        permute_52: "f32[128, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_37: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg120_1, view_104, permute_52);  arg120_1 = view_104 = permute_52 = None
        view_105: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_37, [256, 128, 512]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_8: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_105);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_106: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_8, [32768, 512]);  relu_8 = None
        permute_53: "f32[512, 128]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_38: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg122_1, view_106, permute_53);  arg122_1 = view_106 = permute_53 = None
        view_107: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_38, [256, 128, 128]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_40: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_107, add_39);  view_107 = add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_26: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_40, arg123_1);  add_40 = arg123_1 = None
        add_41: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_26, arg124_1);  mul_26 = arg124_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_41, [32768, 128])
        permute_54: "f32[128, 512]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_39: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg126_1, view_108, permute_54);  arg126_1 = view_108 = permute_54 = None
        view_109: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_39, [256, 128, 512]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_9: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_109);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_110: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_9, [32768, 512]);  relu_9 = None
        permute_55: "f32[512, 128]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_40: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg128_1, view_110, permute_55);  arg128_1 = view_110 = permute_55 = None
        view_111: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_40, [256, 128, 128]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_42: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_111, add_41);  view_111 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_27: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_42, arg129_1);  add_42 = arg129_1 = None
        add_43: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_27, arg130_1);  mul_27 = arg130_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_112: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_43, [32768, 128])
        permute_56: "f32[128, 512]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_41: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg132_1, view_112, permute_56);  arg132_1 = view_112 = permute_56 = None
        view_113: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_41, [256, 128, 512]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_10: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_113);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_114: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_10, [32768, 512]);  relu_10 = None
        permute_57: "f32[512, 128]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_42: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg134_1, view_114, permute_57);  arg134_1 = view_114 = permute_57 = None
        view_115: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_42, [256, 128, 128]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_44: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_115, add_43);  view_115 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_28: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_44, arg135_1);  add_44 = arg135_1 = None
        add_45: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_28, arg136_1);  mul_28 = arg136_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_116: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_45, [32768, 128])
        permute_58: "f32[128, 512]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_43: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg138_1, view_116, permute_58);  arg138_1 = view_116 = permute_58 = None
        view_117: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_43, [256, 128, 512]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_11: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_117);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_118: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_11, [32768, 512]);  relu_11 = None
        permute_59: "f32[512, 128]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_44: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg140_1, view_118, permute_59);  arg140_1 = view_118 = permute_59 = None
        view_119: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_44, [256, 128, 128]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_46: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_119, add_45);  view_119 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_29: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_46, arg141_1);  add_46 = arg141_1 = None
        add_47: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_29, arg142_1);  mul_29 = arg142_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_120: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_47, [32768, 128]);  add_47 = None
        permute_60: "f32[128, 512]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_45: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg144_1, view_120, permute_60);  arg144_1 = view_120 = permute_60 = None
        view_121: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_45, [256, 128, 512]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_48: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_121, add_34);  view_121 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_30: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_48, arg145_1);  add_48 = arg145_1 = None
        add_49: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_30, arg146_1);  mul_30 = arg146_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_124: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_49, [32768, 512])
        permute_62: "f32[512, 128]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_47: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg152_1, view_124, permute_62);  arg152_1 = view_124 = permute_62 = None
        view_125: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_47, [256, 128, 128]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_32: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_125, arg153_1);  view_125 = arg153_1 = None
        add_51: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_32, arg154_1);  mul_32 = arg154_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_126: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_51, [32768, 128])
        permute_63: "f32[128, 128]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_48: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg156_1, view_126, permute_63);  arg156_1 = view_126 = permute_63 = None
        view_127: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_48, [256, 128, 128]);  addmm_48 = None
        view_128: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_127, [256, 128, -1, 32]);  view_127 = None
        permute_64: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_33: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_64, 0.4204482076268573);  permute_64 = None
        expand_13: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_33, [256, 4, 128, 32]);  mul_33 = None
        clone_16: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_135: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_16, [1024, 128, 32]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_129: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_51, [32768, 128]);  add_51 = None
        permute_65: "f32[128, 128]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_49: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg158_1, view_129, permute_65);  arg158_1 = view_129 = permute_65 = None
        view_130: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_49, [256, 128, 128]);  addmm_49 = None
        view_131: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_130, [256, 128, -1, 32]);  view_130 = None
        permute_66: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_69: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_66, [0, 1, 3, 2]);  permute_66 = None
        mul_34: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_69, 0.4204482076268573);  permute_69 = None
        expand_14: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_34, [256, 4, 32, 128]);  mul_34 = None
        clone_17: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_136: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_17, [1024, 32, 128]);  clone_17 = None
        bmm_6: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
        view_137: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [256, 4, 128, 128]);  bmm_6 = None
        full_default_11: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_11, full_default_10);  full_default_11 = full_default_10 = None
        add_52: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_137, where_6);  view_137 = where_6 = None
        eq_3: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_52, -inf)
        logical_not_6: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_12: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_3: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_52, [-1], True)
        sub_3: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_52, amax_3);  add_52 = amax_3 = None
        exp_3: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_4: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        where_7: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_7, full_default_12, div_3);  logical_not_7 = full_default_12 = div_3 = None
        expand_15: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_7, [256, 4, 128, 128]);  where_7 = None
        view_138: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_15, [1024, 128, 128]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_132: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_49, [32768, 512])
        permute_67: "f32[512, 128]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_50: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg160_1, view_132, permute_67);  arg160_1 = view_132 = permute_67 = None
        view_133: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_50, [256, 128, 128]);  addmm_50 = None
        view_134: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_133, [256, 128, -1, 32]);  view_133 = None
        permute_68: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_16: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_68, [256, 4, 128, 32]);  permute_68 = None
        clone_18: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_139: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_18, [1024, 128, 32]);  clone_18 = None
        bmm_7: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_138, view_139);  view_138 = view_139 = None
        view_140: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_7, [256, 4, 128, 32]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_70: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        clone_19: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_141: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_19, [256, 128, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_142: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_141, [32768, 128]);  view_141 = None
        permute_71: "f32[128, 128]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_51: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg162_1, view_142, permute_71);  arg162_1 = view_142 = permute_71 = None
        view_143: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_51, [256, 128, 128]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_122: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_49, [32768, 512])
        permute_61: "f32[512, 128]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_46: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg148_1, view_122, permute_61);  arg148_1 = view_122 = permute_61 = None
        view_123: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_46, [256, 128, 128]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_31: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_123, arg149_1);  view_123 = arg149_1 = None
        add_50: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_31, arg150_1);  mul_31 = arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_53: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_143, add_50);  view_143 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_35: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_53, arg163_1);  add_53 = arg163_1 = None
        add_54: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_35, arg164_1);  mul_35 = arg164_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_144: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_54, [32768, 128])
        permute_72: "f32[128, 512]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_52: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg166_1, view_144, permute_72);  arg166_1 = view_144 = permute_72 = None
        view_145: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_52, [256, 128, 512]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_12: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_145);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_146: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_12, [32768, 512]);  relu_12 = None
        permute_73: "f32[512, 128]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_53: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg168_1, view_146, permute_73);  arg168_1 = view_146 = permute_73 = None
        view_147: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_53, [256, 128, 128]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_55: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_147, add_54);  view_147 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_36: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_55, arg169_1);  add_55 = arg169_1 = None
        add_56: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_36, arg170_1);  mul_36 = arg170_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_56, [32768, 128])
        permute_74: "f32[128, 512]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_54: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg172_1, view_148, permute_74);  arg172_1 = view_148 = permute_74 = None
        view_149: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_54, [256, 128, 512]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_13: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_149);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_150: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_13, [32768, 512]);  relu_13 = None
        permute_75: "f32[512, 128]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_55: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg174_1, view_150, permute_75);  arg174_1 = view_150 = permute_75 = None
        view_151: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_55, [256, 128, 128]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_57: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_151, add_56);  view_151 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_37: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_57, arg175_1);  add_57 = arg175_1 = None
        add_58: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_37, arg176_1);  mul_37 = arg176_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_58, [32768, 128])
        permute_76: "f32[128, 512]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_56: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg178_1, view_152, permute_76);  arg178_1 = view_152 = permute_76 = None
        view_153: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_56, [256, 128, 512]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_14: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_153);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_154: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_14, [32768, 512]);  relu_14 = None
        permute_77: "f32[512, 128]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_57: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg180_1, view_154, permute_77);  arg180_1 = view_154 = permute_77 = None
        view_155: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_57, [256, 128, 128]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_59: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_155, add_58);  view_155 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_38: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_59, arg181_1);  add_59 = arg181_1 = None
        add_60: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_38, arg182_1);  mul_38 = arg182_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_156: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_60, [32768, 128])
        permute_78: "f32[128, 512]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_58: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg184_1, view_156, permute_78);  arg184_1 = view_156 = permute_78 = None
        view_157: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_58, [256, 128, 512]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_15: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_157);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_158: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_15, [32768, 512]);  relu_15 = None
        permute_79: "f32[512, 128]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_59: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg186_1, view_158, permute_79);  arg186_1 = view_158 = permute_79 = None
        view_159: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_59, [256, 128, 128]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_61: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_159, add_60);  view_159 = add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_39: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_61, arg187_1);  add_61 = arg187_1 = None
        add_62: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_39, arg188_1);  mul_39 = arg188_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_160: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_62, [32768, 128]);  add_62 = None
        permute_80: "f32[128, 512]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_60: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg190_1, view_160, permute_80);  arg190_1 = view_160 = permute_80 = None
        view_161: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_60, [256, 128, 512]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_63: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_161, add_49);  view_161 = add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_40: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_63, arg191_1);  add_63 = arg191_1 = None
        add_64: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_40, arg192_1);  mul_40 = arg192_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_164: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_64, [32768, 512])
        permute_82: "f32[512, 128]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_62: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg198_1, view_164, permute_82);  arg198_1 = view_164 = permute_82 = None
        view_165: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_62, [256, 128, 128]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_42: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_165, arg199_1);  view_165 = arg199_1 = None
        add_66: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_42, arg200_1);  mul_42 = arg200_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_166: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_66, [32768, 128])
        permute_83: "f32[128, 128]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_63: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg202_1, view_166, permute_83);  arg202_1 = view_166 = permute_83 = None
        view_167: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_63, [256, 128, 128]);  addmm_63 = None
        view_168: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_167, [256, 128, -1, 32]);  view_167 = None
        permute_84: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_43: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_84, 0.4204482076268573);  permute_84 = None
        expand_17: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_43, [256, 4, 128, 32]);  mul_43 = None
        clone_21: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_175: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_21, [1024, 128, 32]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_169: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_66, [32768, 128]);  add_66 = None
        permute_85: "f32[128, 128]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_64: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg204_1, view_169, permute_85);  arg204_1 = view_169 = permute_85 = None
        view_170: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_64, [256, 128, 128]);  addmm_64 = None
        view_171: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_170, [256, 128, -1, 32]);  view_170 = None
        permute_86: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_89: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_86, [0, 1, 3, 2]);  permute_86 = None
        mul_44: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_89, 0.4204482076268573);  permute_89 = None
        expand_18: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_44, [256, 4, 32, 128]);  mul_44 = None
        clone_22: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_176: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_22, [1024, 32, 128]);  clone_22 = None
        bmm_8: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_175, view_176);  view_175 = view_176 = None
        view_177: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [256, 4, 128, 128]);  bmm_8 = None
        full_default_14: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        add_67: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_177, where_8);  view_177 = where_8 = None
        eq_4: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_67, -inf)
        logical_not_8: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_15: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_4: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_67, [-1], True)
        sub_4: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_67, amax_4);  add_67 = amax_4 = None
        exp_4: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_5: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        where_9: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_9, full_default_15, div_4);  logical_not_9 = full_default_15 = div_4 = None
        expand_19: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_9, [256, 4, 128, 128]);  where_9 = None
        view_178: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_19, [1024, 128, 128]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_172: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_64, [32768, 512])
        permute_87: "f32[512, 128]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_65: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg206_1, view_172, permute_87);  arg206_1 = view_172 = permute_87 = None
        view_173: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_65, [256, 128, 128]);  addmm_65 = None
        view_174: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_173, [256, 128, -1, 32]);  view_173 = None
        permute_88: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_20: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_88, [256, 4, 128, 32]);  permute_88 = None
        clone_23: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_179: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_23, [1024, 128, 32]);  clone_23 = None
        bmm_9: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_178, view_179);  view_178 = view_179 = None
        view_180: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_9, [256, 4, 128, 32]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_90: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        clone_24: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_181: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_24, [256, 128, -1]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_182: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_181, [32768, 128]);  view_181 = None
        permute_91: "f32[128, 128]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_66: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg208_1, view_182, permute_91);  arg208_1 = view_182 = permute_91 = None
        view_183: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_66, [256, 128, 128]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_162: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_64, [32768, 512])
        permute_81: "f32[512, 128]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_61: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg194_1, view_162, permute_81);  arg194_1 = view_162 = permute_81 = None
        view_163: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_61, [256, 128, 128]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_41: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_163, arg195_1);  view_163 = arg195_1 = None
        add_65: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_41, arg196_1);  mul_41 = arg196_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_68: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_183, add_65);  view_183 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_45: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_68, arg209_1);  add_68 = arg209_1 = None
        add_69: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_45, arg210_1);  mul_45 = arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_184: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_69, [32768, 128])
        permute_92: "f32[128, 512]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_67: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg212_1, view_184, permute_92);  arg212_1 = view_184 = permute_92 = None
        view_185: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_67, [256, 128, 512]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_16: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_185);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_186: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_16, [32768, 512]);  relu_16 = None
        permute_93: "f32[512, 128]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_68: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg214_1, view_186, permute_93);  arg214_1 = view_186 = permute_93 = None
        view_187: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_68, [256, 128, 128]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_70: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_187, add_69);  view_187 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_46: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_70, arg215_1);  add_70 = arg215_1 = None
        add_71: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, arg216_1);  mul_46 = arg216_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_188: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_71, [32768, 128])
        permute_94: "f32[128, 512]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_69: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg218_1, view_188, permute_94);  arg218_1 = view_188 = permute_94 = None
        view_189: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_69, [256, 128, 512]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_17: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_189);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_190: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_17, [32768, 512]);  relu_17 = None
        permute_95: "f32[512, 128]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_70: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg220_1, view_190, permute_95);  arg220_1 = view_190 = permute_95 = None
        view_191: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_70, [256, 128, 128]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_72: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_191, add_71);  view_191 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_47: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_72, arg221_1);  add_72 = arg221_1 = None
        add_73: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_47, arg222_1);  mul_47 = arg222_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_73, [32768, 128])
        permute_96: "f32[128, 512]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_71: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg224_1, view_192, permute_96);  arg224_1 = view_192 = permute_96 = None
        view_193: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_71, [256, 128, 512]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_18: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_193);  view_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_194: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_18, [32768, 512]);  relu_18 = None
        permute_97: "f32[512, 128]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_72: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg226_1, view_194, permute_97);  arg226_1 = view_194 = permute_97 = None
        view_195: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_72, [256, 128, 128]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_74: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_195, add_73);  view_195 = add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_48: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_74, arg227_1);  add_74 = arg227_1 = None
        add_75: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_48, arg228_1);  mul_48 = arg228_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_75, [32768, 128])
        permute_98: "f32[128, 512]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_73: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg230_1, view_196, permute_98);  arg230_1 = view_196 = permute_98 = None
        view_197: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_73, [256, 128, 512]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_19: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_197);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_198: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_19, [32768, 512]);  relu_19 = None
        permute_99: "f32[512, 128]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_74: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg232_1, view_198, permute_99);  arg232_1 = view_198 = permute_99 = None
        view_199: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_74, [256, 128, 128]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_76: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_199, add_75);  view_199 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_49: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_76, arg233_1);  add_76 = arg233_1 = None
        add_77: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_49, arg234_1);  mul_49 = arg234_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_200: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_77, [32768, 128]);  add_77 = None
        permute_100: "f32[128, 512]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_75: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg236_1, view_200, permute_100);  arg236_1 = view_200 = permute_100 = None
        view_201: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_75, [256, 128, 512]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_78: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_201, add_64);  view_201 = add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_50: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_78, arg237_1);  add_78 = arg237_1 = None
        add_79: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_50, arg238_1);  mul_50 = arg238_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_204: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_79, [32768, 512])
        permute_102: "f32[512, 128]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_77: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg244_1, view_204, permute_102);  arg244_1 = view_204 = permute_102 = None
        view_205: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_77, [256, 128, 128]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_52: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_205, arg245_1);  view_205 = arg245_1 = None
        add_81: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_52, arg246_1);  mul_52 = arg246_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_206: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_81, [32768, 128])
        permute_103: "f32[128, 128]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_78: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg248_1, view_206, permute_103);  arg248_1 = view_206 = permute_103 = None
        view_207: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_78, [256, 128, 128]);  addmm_78 = None
        view_208: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_207, [256, 128, -1, 32]);  view_207 = None
        permute_104: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_53: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_104, 0.4204482076268573);  permute_104 = None
        expand_21: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_53, [256, 4, 128, 32]);  mul_53 = None
        clone_26: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_215: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_26, [1024, 128, 32]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_209: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_81, [32768, 128]);  add_81 = None
        permute_105: "f32[128, 128]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_79: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg250_1, view_209, permute_105);  arg250_1 = view_209 = permute_105 = None
        view_210: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_79, [256, 128, 128]);  addmm_79 = None
        view_211: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_210, [256, 128, -1, 32]);  view_210 = None
        permute_106: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_109: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_106, [0, 1, 3, 2]);  permute_106 = None
        mul_54: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_109, 0.4204482076268573);  permute_109 = None
        expand_22: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_54, [256, 4, 32, 128]);  mul_54 = None
        clone_27: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_216: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_27, [1024, 32, 128]);  clone_27 = None
        bmm_10: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_215, view_216);  view_215 = view_216 = None
        view_217: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [256, 4, 128, 128]);  bmm_10 = None
        full_default_17: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        add_82: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_217, where_10);  view_217 = where_10 = None
        eq_5: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_82, -inf)
        logical_not_10: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_18: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_5: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_82, [-1], True)
        sub_5: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_82, amax_5);  add_82 = amax_5 = None
        exp_5: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_6: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        where_11: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_11, full_default_18, div_5);  logical_not_11 = full_default_18 = div_5 = None
        expand_23: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_11, [256, 4, 128, 128]);  where_11 = None
        view_218: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_23, [1024, 128, 128]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_212: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_79, [32768, 512])
        permute_107: "f32[512, 128]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_80: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg252_1, view_212, permute_107);  arg252_1 = view_212 = permute_107 = None
        view_213: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_80, [256, 128, 128]);  addmm_80 = None
        view_214: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_213, [256, 128, -1, 32]);  view_213 = None
        permute_108: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_24: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_108, [256, 4, 128, 32]);  permute_108 = None
        clone_28: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_219: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_28, [1024, 128, 32]);  clone_28 = None
        bmm_11: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_218, view_219);  view_218 = view_219 = None
        view_220: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_11, [256, 4, 128, 32]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_110: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
        clone_29: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_221: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_29, [256, 128, -1]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_222: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_221, [32768, 128]);  view_221 = None
        permute_111: "f32[128, 128]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_81: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg254_1, view_222, permute_111);  arg254_1 = view_222 = permute_111 = None
        view_223: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_81, [256, 128, 128]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_202: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_79, [32768, 512])
        permute_101: "f32[512, 128]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_76: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg240_1, view_202, permute_101);  arg240_1 = view_202 = permute_101 = None
        view_203: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_76, [256, 128, 128]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_51: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_203, arg241_1);  view_203 = arg241_1 = None
        add_80: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_51, arg242_1);  mul_51 = arg242_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_83: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_223, add_80);  view_223 = add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_55: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_83, arg255_1);  add_83 = arg255_1 = None
        add_84: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_55, arg256_1);  mul_55 = arg256_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_224: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_84, [32768, 128])
        permute_112: "f32[128, 512]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_82: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg258_1, view_224, permute_112);  arg258_1 = view_224 = permute_112 = None
        view_225: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_82, [256, 128, 512]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_20: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_225);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_226: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_20, [32768, 512]);  relu_20 = None
        permute_113: "f32[512, 128]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_83: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg260_1, view_226, permute_113);  arg260_1 = view_226 = permute_113 = None
        view_227: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_83, [256, 128, 128]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_85: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_227, add_84);  view_227 = add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_56: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_85, arg261_1);  add_85 = arg261_1 = None
        add_86: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_56, arg262_1);  mul_56 = arg262_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_228: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_86, [32768, 128])
        permute_114: "f32[128, 512]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_84: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg264_1, view_228, permute_114);  arg264_1 = view_228 = permute_114 = None
        view_229: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_84, [256, 128, 512]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_21: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_229);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_230: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_21, [32768, 512]);  relu_21 = None
        permute_115: "f32[512, 128]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_85: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg266_1, view_230, permute_115);  arg266_1 = view_230 = permute_115 = None
        view_231: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_85, [256, 128, 128]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_87: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_231, add_86);  view_231 = add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_57: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_87, arg267_1);  add_87 = arg267_1 = None
        add_88: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_57, arg268_1);  mul_57 = arg268_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_232: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_88, [32768, 128])
        permute_116: "f32[128, 512]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_86: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg270_1, view_232, permute_116);  arg270_1 = view_232 = permute_116 = None
        view_233: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_86, [256, 128, 512]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_22: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_233);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_234: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_22, [32768, 512]);  relu_22 = None
        permute_117: "f32[512, 128]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_87: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg272_1, view_234, permute_117);  arg272_1 = view_234 = permute_117 = None
        view_235: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_87, [256, 128, 128]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_89: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_235, add_88);  view_235 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_58: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_89, arg273_1);  add_89 = arg273_1 = None
        add_90: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_58, arg274_1);  mul_58 = arg274_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_90, [32768, 128])
        permute_118: "f32[128, 512]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_88: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg276_1, view_236, permute_118);  arg276_1 = view_236 = permute_118 = None
        view_237: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_88, [256, 128, 512]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_23: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_237);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_238: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_23, [32768, 512]);  relu_23 = None
        permute_119: "f32[512, 128]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_89: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg278_1, view_238, permute_119);  arg278_1 = view_238 = permute_119 = None
        view_239: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_89, [256, 128, 128]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_91: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_239, add_90);  view_239 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_59: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_91, arg279_1);  add_91 = arg279_1 = None
        add_92: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_59, arg280_1);  mul_59 = arg280_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_240: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_92, [32768, 128]);  add_92 = None
        permute_120: "f32[128, 512]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_90: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg282_1, view_240, permute_120);  arg282_1 = view_240 = permute_120 = None
        view_241: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_90, [256, 128, 512]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_93: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_241, add_79);  view_241 = add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_60: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_93, arg283_1);  add_93 = arg283_1 = None
        add_94: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_60, arg284_1);  mul_60 = arg284_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_244: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_94, [32768, 512])
        permute_122: "f32[512, 128]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_92: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg290_1, view_244, permute_122);  arg290_1 = view_244 = permute_122 = None
        view_245: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_92, [256, 128, 128]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_62: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_245, arg291_1);  view_245 = arg291_1 = None
        add_96: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_62, arg292_1);  mul_62 = arg292_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_246: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_96, [32768, 128])
        permute_123: "f32[128, 128]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_93: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg294_1, view_246, permute_123);  arg294_1 = view_246 = permute_123 = None
        view_247: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_93, [256, 128, 128]);  addmm_93 = None
        view_248: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_247, [256, 128, -1, 32]);  view_247 = None
        permute_124: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_63: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_124, 0.4204482076268573);  permute_124 = None
        expand_25: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_63, [256, 4, 128, 32]);  mul_63 = None
        clone_31: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_255: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_31, [1024, 128, 32]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_249: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_96, [32768, 128]);  add_96 = None
        permute_125: "f32[128, 128]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_94: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg296_1, view_249, permute_125);  arg296_1 = view_249 = permute_125 = None
        view_250: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_94, [256, 128, 128]);  addmm_94 = None
        view_251: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_250, [256, 128, -1, 32]);  view_250 = None
        permute_126: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_129: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_126, [0, 1, 3, 2]);  permute_126 = None
        mul_64: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_129, 0.4204482076268573);  permute_129 = None
        expand_26: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_64, [256, 4, 32, 128]);  mul_64 = None
        clone_32: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_256: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_32, [1024, 32, 128]);  clone_32 = None
        bmm_12: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_255, view_256);  view_255 = view_256 = None
        view_257: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [256, 4, 128, 128]);  bmm_12 = None
        full_default_20: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        add_97: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_257, where_12);  view_257 = where_12 = None
        eq_6: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_97, -inf)
        logical_not_12: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_default_21: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_6: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_97, [-1], True)
        sub_6: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_97, amax_6);  add_97 = amax_6 = None
        exp_6: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_7: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        where_13: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_13, full_default_21, div_6);  logical_not_13 = full_default_21 = div_6 = None
        expand_27: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_13, [256, 4, 128, 128]);  where_13 = None
        view_258: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_27, [1024, 128, 128]);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_252: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_94, [32768, 512])
        permute_127: "f32[512, 128]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_95: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg298_1, view_252, permute_127);  arg298_1 = view_252 = permute_127 = None
        view_253: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_95, [256, 128, 128]);  addmm_95 = None
        view_254: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_253, [256, 128, -1, 32]);  view_253 = None
        permute_128: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_28: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_128, [256, 4, 128, 32]);  permute_128 = None
        clone_33: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_259: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_33, [1024, 128, 32]);  clone_33 = None
        bmm_13: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_258, view_259);  view_258 = view_259 = None
        view_260: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_13, [256, 4, 128, 32]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_130: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
        clone_34: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_261: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_34, [256, 128, -1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_262: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_261, [32768, 128]);  view_261 = None
        permute_131: "f32[128, 128]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_96: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg300_1, view_262, permute_131);  arg300_1 = view_262 = permute_131 = None
        view_263: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_96, [256, 128, 128]);  addmm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_242: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_94, [32768, 512])
        permute_121: "f32[512, 128]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_91: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg286_1, view_242, permute_121);  arg286_1 = view_242 = permute_121 = None
        view_243: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_91, [256, 128, 128]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_61: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_243, arg287_1);  view_243 = arg287_1 = None
        add_95: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_61, arg288_1);  mul_61 = arg288_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_98: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_263, add_95);  view_263 = add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_65: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_98, arg301_1);  add_98 = arg301_1 = None
        add_99: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_65, arg302_1);  mul_65 = arg302_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_264: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_99, [32768, 128])
        permute_132: "f32[128, 512]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_97: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg304_1, view_264, permute_132);  arg304_1 = view_264 = permute_132 = None
        view_265: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_97, [256, 128, 512]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_24: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_265);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_266: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_24, [32768, 512]);  relu_24 = None
        permute_133: "f32[512, 128]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_98: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg306_1, view_266, permute_133);  arg306_1 = view_266 = permute_133 = None
        view_267: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_98, [256, 128, 128]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_100: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_267, add_99);  view_267 = add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_66: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_100, arg307_1);  add_100 = arg307_1 = None
        add_101: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_66, arg308_1);  mul_66 = arg308_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_268: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_101, [32768, 128])
        permute_134: "f32[128, 512]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_99: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg310_1, view_268, permute_134);  arg310_1 = view_268 = permute_134 = None
        view_269: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_99, [256, 128, 512]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_25: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_269);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_270: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_25, [32768, 512]);  relu_25 = None
        permute_135: "f32[512, 128]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_100: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg312_1, view_270, permute_135);  arg312_1 = view_270 = permute_135 = None
        view_271: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_100, [256, 128, 128]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_102: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_271, add_101);  view_271 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_67: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_102, arg313_1);  add_102 = arg313_1 = None
        add_103: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_67, arg314_1);  mul_67 = arg314_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_272: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_103, [32768, 128])
        permute_136: "f32[128, 512]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_101: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg316_1, view_272, permute_136);  arg316_1 = view_272 = permute_136 = None
        view_273: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_101, [256, 128, 512]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_26: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_273);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_274: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_26, [32768, 512]);  relu_26 = None
        permute_137: "f32[512, 128]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_102: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg318_1, view_274, permute_137);  arg318_1 = view_274 = permute_137 = None
        view_275: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_102, [256, 128, 128]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_104: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_275, add_103);  view_275 = add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_68: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_104, arg319_1);  add_104 = arg319_1 = None
        add_105: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_68, arg320_1);  mul_68 = arg320_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_276: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_105, [32768, 128])
        permute_138: "f32[128, 512]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_103: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg322_1, view_276, permute_138);  arg322_1 = view_276 = permute_138 = None
        view_277: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_103, [256, 128, 512]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_27: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_277);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_278: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_27, [32768, 512]);  relu_27 = None
        permute_139: "f32[512, 128]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_104: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg324_1, view_278, permute_139);  arg324_1 = view_278 = permute_139 = None
        view_279: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_104, [256, 128, 128]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_106: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_279, add_105);  view_279 = add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_69: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_106, arg325_1);  add_106 = arg325_1 = None
        add_107: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_69, arg326_1);  mul_69 = arg326_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_280: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_107, [32768, 128]);  add_107 = None
        permute_140: "f32[128, 512]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_105: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg328_1, view_280, permute_140);  arg328_1 = view_280 = permute_140 = None
        view_281: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_105, [256, 128, 512]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_108: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_281, add_94);  view_281 = add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_70: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_108, arg329_1);  add_108 = arg329_1 = None
        add_109: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_70, arg330_1);  mul_70 = arg330_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_284: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_109, [32768, 512])
        permute_142: "f32[512, 128]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_107: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg336_1, view_284, permute_142);  arg336_1 = view_284 = permute_142 = None
        view_285: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_107, [256, 128, 128]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_72: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_285, arg337_1);  view_285 = arg337_1 = None
        add_111: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_72, arg338_1);  mul_72 = arg338_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_286: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_111, [32768, 128])
        permute_143: "f32[128, 128]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_108: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg340_1, view_286, permute_143);  arg340_1 = view_286 = permute_143 = None
        view_287: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_108, [256, 128, 128]);  addmm_108 = None
        view_288: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_287, [256, 128, -1, 32]);  view_287 = None
        permute_144: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_73: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_144, 0.4204482076268573);  permute_144 = None
        expand_29: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_73, [256, 4, 128, 32]);  mul_73 = None
        clone_36: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_295: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_36, [1024, 128, 32]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_289: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_111, [32768, 128]);  add_111 = None
        permute_145: "f32[128, 128]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_109: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg342_1, view_289, permute_145);  arg342_1 = view_289 = permute_145 = None
        view_290: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_109, [256, 128, 128]);  addmm_109 = None
        view_291: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_290, [256, 128, -1, 32]);  view_290 = None
        permute_146: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_149: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_146, [0, 1, 3, 2]);  permute_146 = None
        mul_74: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_149, 0.4204482076268573);  permute_149 = None
        expand_30: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_74, [256, 4, 32, 128]);  mul_74 = None
        clone_37: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_296: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_37, [1024, 32, 128]);  clone_37 = None
        bmm_14: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_295, view_296);  view_295 = view_296 = None
        view_297: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [256, 4, 128, 128]);  bmm_14 = None
        full_default_23: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_23, full_default_22);  full_default_23 = full_default_22 = None
        add_112: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_297, where_14);  view_297 = where_14 = None
        eq_7: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_112, -inf)
        logical_not_14: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_default_24: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_7: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_112, [-1], True)
        sub_7: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_112, amax_7);  add_112 = amax_7 = None
        exp_7: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_8: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        where_15: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_15, full_default_24, div_7);  logical_not_15 = full_default_24 = div_7 = None
        expand_31: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_15, [256, 4, 128, 128]);  where_15 = None
        view_298: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_31, [1024, 128, 128]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_292: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_109, [32768, 512])
        permute_147: "f32[512, 128]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_110: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg344_1, view_292, permute_147);  arg344_1 = view_292 = permute_147 = None
        view_293: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_110, [256, 128, 128]);  addmm_110 = None
        view_294: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_293, [256, 128, -1, 32]);  view_293 = None
        permute_148: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_32: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_148, [256, 4, 128, 32]);  permute_148 = None
        clone_38: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_299: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_38, [1024, 128, 32]);  clone_38 = None
        bmm_15: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_298, view_299);  view_298 = view_299 = None
        view_300: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_15, [256, 4, 128, 32]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_39: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_301: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_39, [256, 128, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_302: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_301, [32768, 128]);  view_301 = None
        permute_151: "f32[128, 128]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_111: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg346_1, view_302, permute_151);  arg346_1 = view_302 = permute_151 = None
        view_303: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_111, [256, 128, 128]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_282: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_109, [32768, 512])
        permute_141: "f32[512, 128]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_106: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg332_1, view_282, permute_141);  arg332_1 = view_282 = permute_141 = None
        view_283: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_106, [256, 128, 128]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_71: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_283, arg333_1);  view_283 = arg333_1 = None
        add_110: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_71, arg334_1);  mul_71 = arg334_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_113: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_303, add_110);  view_303 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_75: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_113, arg347_1);  add_113 = arg347_1 = None
        add_114: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_75, arg348_1);  mul_75 = arg348_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_304: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_114, [32768, 128])
        permute_152: "f32[128, 512]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_112: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg350_1, view_304, permute_152);  arg350_1 = view_304 = permute_152 = None
        view_305: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_112, [256, 128, 512]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_28: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_305);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_306: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_28, [32768, 512]);  relu_28 = None
        permute_153: "f32[512, 128]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_113: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg352_1, view_306, permute_153);  arg352_1 = view_306 = permute_153 = None
        view_307: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_113, [256, 128, 128]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_115: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_307, add_114);  view_307 = add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_76: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_115, arg353_1);  add_115 = arg353_1 = None
        add_116: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_76, arg354_1);  mul_76 = arg354_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_308: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_116, [32768, 128])
        permute_154: "f32[128, 512]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        addmm_114: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg356_1, view_308, permute_154);  arg356_1 = view_308 = permute_154 = None
        view_309: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_114, [256, 128, 512]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_29: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_309);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_310: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_29, [32768, 512]);  relu_29 = None
        permute_155: "f32[512, 128]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_115: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg358_1, view_310, permute_155);  arg358_1 = view_310 = permute_155 = None
        view_311: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_115, [256, 128, 128]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_117: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_311, add_116);  view_311 = add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_77: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_117, arg359_1);  add_117 = arg359_1 = None
        add_118: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_77, arg360_1);  mul_77 = arg360_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_312: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_118, [32768, 128])
        permute_156: "f32[128, 512]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_116: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg362_1, view_312, permute_156);  arg362_1 = view_312 = permute_156 = None
        view_313: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_116, [256, 128, 512]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_30: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_313);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_314: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_30, [32768, 512]);  relu_30 = None
        permute_157: "f32[512, 128]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_117: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg364_1, view_314, permute_157);  arg364_1 = view_314 = permute_157 = None
        view_315: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_117, [256, 128, 128]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_119: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_315, add_118);  view_315 = add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_78: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_119, arg365_1);  add_119 = arg365_1 = None
        add_120: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_78, arg366_1);  mul_78 = arg366_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_316: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_120, [32768, 128])
        permute_158: "f32[128, 512]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_118: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg368_1, view_316, permute_158);  arg368_1 = view_316 = permute_158 = None
        view_317: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_118, [256, 128, 512]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_31: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_317);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_318: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_31, [32768, 512]);  relu_31 = None
        permute_159: "f32[512, 128]" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_119: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg370_1, view_318, permute_159);  arg370_1 = view_318 = permute_159 = None
        view_319: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_119, [256, 128, 128]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_121: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_319, add_120);  view_319 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_79: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_121, arg371_1);  add_121 = arg371_1 = None
        add_122: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_79, arg372_1);  mul_79 = arg372_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_320: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_122, [32768, 128]);  add_122 = None
        permute_160: "f32[128, 512]" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_120: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg374_1, view_320, permute_160);  arg374_1 = view_320 = permute_160 = None
        view_321: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_120, [256, 128, 512]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_123: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_321, add_109);  view_321 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_80: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_123, arg375_1);  add_123 = arg375_1 = None
        add_124: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_80, arg376_1);  mul_80 = arg376_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_324: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_124, [32768, 512])
        permute_162: "f32[512, 128]" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
        addmm_122: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg382_1, view_324, permute_162);  arg382_1 = view_324 = permute_162 = None
        view_325: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_122, [256, 128, 128]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_82: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_325, arg383_1);  view_325 = arg383_1 = None
        add_126: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_82, arg384_1);  mul_82 = arg384_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_326: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_126, [32768, 128])
        permute_163: "f32[128, 128]" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_123: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg386_1, view_326, permute_163);  arg386_1 = view_326 = permute_163 = None
        view_327: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_123, [256, 128, 128]);  addmm_123 = None
        view_328: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_327, [256, 128, -1, 32]);  view_327 = None
        permute_164: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_83: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_164, 0.4204482076268573);  permute_164 = None
        expand_33: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_83, [256, 4, 128, 32]);  mul_83 = None
        clone_41: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_335: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_41, [1024, 128, 32]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_329: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_126, [32768, 128]);  add_126 = None
        permute_165: "f32[128, 128]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        addmm_124: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg388_1, view_329, permute_165);  arg388_1 = view_329 = permute_165 = None
        view_330: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_124, [256, 128, 128]);  addmm_124 = None
        view_331: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_330, [256, 128, -1, 32]);  view_330 = None
        permute_166: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_169: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_166, [0, 1, 3, 2]);  permute_166 = None
        mul_84: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_169, 0.4204482076268573);  permute_169 = None
        expand_34: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_84, [256, 4, 32, 128]);  mul_84 = None
        clone_42: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_336: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_42, [1024, 32, 128]);  clone_42 = None
        bmm_16: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_335, view_336);  view_335 = view_336 = None
        view_337: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [256, 4, 128, 128]);  bmm_16 = None
        full_default_26: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_26, full_default_25);  full_default_26 = full_default_25 = None
        add_127: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_337, where_16);  view_337 = where_16 = None
        eq_8: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_127, -inf)
        logical_not_16: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_default_27: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_8: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_127, [-1], True)
        sub_8: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_127, amax_8);  add_127 = amax_8 = None
        exp_8: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_9: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        where_17: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_17, full_default_27, div_8);  logical_not_17 = full_default_27 = div_8 = None
        expand_35: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_17, [256, 4, 128, 128]);  where_17 = None
        view_338: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_35, [1024, 128, 128]);  expand_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_332: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_124, [32768, 512])
        permute_167: "f32[512, 128]" = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        addmm_125: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg390_1, view_332, permute_167);  arg390_1 = view_332 = permute_167 = None
        view_333: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_125, [256, 128, 128]);  addmm_125 = None
        view_334: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_333, [256, 128, -1, 32]);  view_333 = None
        permute_168: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_36: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_168, [256, 4, 128, 32]);  permute_168 = None
        clone_43: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_339: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_43, [1024, 128, 32]);  clone_43 = None
        bmm_17: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_338, view_339);  view_338 = view_339 = None
        view_340: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_17, [256, 4, 128, 32]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_170: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        clone_44: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_341: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_44, [256, 128, -1]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_342: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_341, [32768, 128]);  view_341 = None
        permute_171: "f32[128, 128]" = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
        addmm_126: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg392_1, view_342, permute_171);  arg392_1 = view_342 = permute_171 = None
        view_343: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_126, [256, 128, 128]);  addmm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_322: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_124, [32768, 512])
        permute_161: "f32[512, 128]" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_121: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg378_1, view_322, permute_161);  arg378_1 = view_322 = permute_161 = None
        view_323: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_121, [256, 128, 128]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_81: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_323, arg379_1);  view_323 = arg379_1 = None
        add_125: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_81, arg380_1);  mul_81 = arg380_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_128: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_343, add_125);  view_343 = add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_85: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_128, arg393_1);  add_128 = arg393_1 = None
        add_129: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_85, arg394_1);  mul_85 = arg394_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_344: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_129, [32768, 128])
        permute_172: "f32[128, 512]" = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
        addmm_127: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg396_1, view_344, permute_172);  arg396_1 = view_344 = permute_172 = None
        view_345: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_127, [256, 128, 512]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_32: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_345);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_346: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_32, [32768, 512]);  relu_32 = None
        permute_173: "f32[512, 128]" = torch.ops.aten.permute.default(arg397_1, [1, 0]);  arg397_1 = None
        addmm_128: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg398_1, view_346, permute_173);  arg398_1 = view_346 = permute_173 = None
        view_347: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_128, [256, 128, 128]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_130: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_347, add_129);  view_347 = add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_86: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_130, arg399_1);  add_130 = arg399_1 = None
        add_131: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_86, arg400_1);  mul_86 = arg400_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_348: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_131, [32768, 128])
        permute_174: "f32[128, 512]" = torch.ops.aten.permute.default(arg401_1, [1, 0]);  arg401_1 = None
        addmm_129: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg402_1, view_348, permute_174);  arg402_1 = view_348 = permute_174 = None
        view_349: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_129, [256, 128, 512]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_33: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_349);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_350: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_33, [32768, 512]);  relu_33 = None
        permute_175: "f32[512, 128]" = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
        addmm_130: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg404_1, view_350, permute_175);  arg404_1 = view_350 = permute_175 = None
        view_351: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_130, [256, 128, 128]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_132: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_351, add_131);  view_351 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_87: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_132, arg405_1);  add_132 = arg405_1 = None
        add_133: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_87, arg406_1);  mul_87 = arg406_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_352: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_133, [32768, 128])
        permute_176: "f32[128, 512]" = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
        addmm_131: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg408_1, view_352, permute_176);  arg408_1 = view_352 = permute_176 = None
        view_353: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_131, [256, 128, 512]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_34: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_353);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_354: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_34, [32768, 512]);  relu_34 = None
        permute_177: "f32[512, 128]" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        addmm_132: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg410_1, view_354, permute_177);  arg410_1 = view_354 = permute_177 = None
        view_355: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_132, [256, 128, 128]);  addmm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_134: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_355, add_133);  view_355 = add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_88: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_134, arg411_1);  add_134 = arg411_1 = None
        add_135: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_88, arg412_1);  mul_88 = arg412_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_356: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_135, [32768, 128])
        permute_178: "f32[128, 512]" = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
        addmm_133: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg414_1, view_356, permute_178);  arg414_1 = view_356 = permute_178 = None
        view_357: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_133, [256, 128, 512]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_35: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_357);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_358: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_35, [32768, 512]);  relu_35 = None
        permute_179: "f32[512, 128]" = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
        addmm_134: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg416_1, view_358, permute_179);  arg416_1 = view_358 = permute_179 = None
        view_359: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_134, [256, 128, 128]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_136: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_359, add_135);  view_359 = add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_89: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_136, arg417_1);  add_136 = arg417_1 = None
        add_137: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_89, arg418_1);  mul_89 = arg418_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_360: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_137, [32768, 128]);  add_137 = None
        permute_180: "f32[128, 512]" = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
        addmm_135: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg420_1, view_360, permute_180);  arg420_1 = view_360 = permute_180 = None
        view_361: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_135, [256, 128, 512]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_138: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_361, add_124);  view_361 = add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_90: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_138, arg421_1);  add_138 = arg421_1 = None
        add_139: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_90, arg422_1);  mul_90 = arg422_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_364: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_139, [32768, 512])
        permute_182: "f32[512, 128]" = torch.ops.aten.permute.default(arg427_1, [1, 0]);  arg427_1 = None
        addmm_137: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg428_1, view_364, permute_182);  arg428_1 = view_364 = permute_182 = None
        view_365: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_137, [256, 128, 128]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_92: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_365, arg429_1);  view_365 = arg429_1 = None
        add_141: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_92, arg430_1);  mul_92 = arg430_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_366: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_141, [32768, 128])
        permute_183: "f32[128, 128]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        addmm_138: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg432_1, view_366, permute_183);  arg432_1 = view_366 = permute_183 = None
        view_367: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_138, [256, 128, 128]);  addmm_138 = None
        view_368: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_367, [256, 128, -1, 32]);  view_367 = None
        permute_184: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_93: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_184, 0.4204482076268573);  permute_184 = None
        expand_37: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_93, [256, 4, 128, 32]);  mul_93 = None
        clone_46: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_375: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_46, [1024, 128, 32]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_369: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_141, [32768, 128]);  add_141 = None
        permute_185: "f32[128, 128]" = torch.ops.aten.permute.default(arg433_1, [1, 0]);  arg433_1 = None
        addmm_139: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg434_1, view_369, permute_185);  arg434_1 = view_369 = permute_185 = None
        view_370: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_139, [256, 128, 128]);  addmm_139 = None
        view_371: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_370, [256, 128, -1, 32]);  view_370 = None
        permute_186: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_189: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_186, [0, 1, 3, 2]);  permute_186 = None
        mul_94: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_189, 0.4204482076268573);  permute_189 = None
        expand_38: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_94, [256, 4, 32, 128]);  mul_94 = None
        clone_47: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_376: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_47, [1024, 32, 128]);  clone_47 = None
        bmm_18: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_375, view_376);  view_375 = view_376 = None
        view_377: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [256, 4, 128, 128]);  bmm_18 = None
        full_default_29: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_29, full_default_28);  full_default_29 = full_default_28 = None
        add_142: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_377, where_18);  view_377 = where_18 = None
        eq_9: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_142, -inf)
        logical_not_18: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_default_30: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_9: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_142, [-1], True)
        sub_9: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_142, amax_9);  add_142 = amax_9 = None
        exp_9: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_10: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        where_19: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_19, full_default_30, div_9);  logical_not_19 = full_default_30 = div_9 = None
        expand_39: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_19, [256, 4, 128, 128]);  where_19 = None
        view_378: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_39, [1024, 128, 128]);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_372: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_139, [32768, 512])
        permute_187: "f32[512, 128]" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_140: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg436_1, view_372, permute_187);  arg436_1 = view_372 = permute_187 = None
        view_373: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_140, [256, 128, 128]);  addmm_140 = None
        view_374: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_373, [256, 128, -1, 32]);  view_373 = None
        permute_188: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_40: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_188, [256, 4, 128, 32]);  permute_188 = None
        clone_48: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_379: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_48, [1024, 128, 32]);  clone_48 = None
        bmm_19: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_378, view_379);  view_378 = view_379 = None
        view_380: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_19, [256, 4, 128, 32]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_190: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
        clone_49: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_381: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_49, [256, 128, -1]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_382: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_381, [32768, 128]);  view_381 = None
        permute_191: "f32[128, 128]" = torch.ops.aten.permute.default(arg437_1, [1, 0]);  arg437_1 = None
        addmm_141: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg438_1, view_382, permute_191);  arg438_1 = view_382 = permute_191 = None
        view_383: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_141, [256, 128, 128]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_362: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_139, [32768, 512])
        permute_181: "f32[512, 128]" = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        addmm_136: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg424_1, view_362, permute_181);  arg424_1 = view_362 = permute_181 = None
        view_363: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_136, [256, 128, 128]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_91: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_363, arg425_1);  view_363 = arg425_1 = None
        add_140: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_91, arg426_1);  mul_91 = arg426_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_143: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_383, add_140);  view_383 = add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_95: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_143, arg439_1);  add_143 = arg439_1 = None
        add_144: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_95, arg440_1);  mul_95 = arg440_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_384: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_144, [32768, 128])
        permute_192: "f32[128, 512]" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_142: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg442_1, view_384, permute_192);  arg442_1 = view_384 = permute_192 = None
        view_385: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_142, [256, 128, 512]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_36: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_385);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_386: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_36, [32768, 512]);  relu_36 = None
        permute_193: "f32[512, 128]" = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
        addmm_143: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg444_1, view_386, permute_193);  arg444_1 = view_386 = permute_193 = None
        view_387: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_143, [256, 128, 128]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_145: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_387, add_144);  view_387 = add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_96: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_145, arg445_1);  add_145 = arg445_1 = None
        add_146: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_96, arg446_1);  mul_96 = arg446_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_388: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_146, [32768, 128])
        permute_194: "f32[128, 512]" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        addmm_144: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg448_1, view_388, permute_194);  arg448_1 = view_388 = permute_194 = None
        view_389: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_144, [256, 128, 512]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_37: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_389);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_390: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_37, [32768, 512]);  relu_37 = None
        permute_195: "f32[512, 128]" = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
        addmm_145: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg450_1, view_390, permute_195);  arg450_1 = view_390 = permute_195 = None
        view_391: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_145, [256, 128, 128]);  addmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_147: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_391, add_146);  view_391 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_97: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_147, arg451_1);  add_147 = arg451_1 = None
        add_148: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_97, arg452_1);  mul_97 = arg452_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_148, [32768, 128])
        permute_196: "f32[128, 512]" = torch.ops.aten.permute.default(arg453_1, [1, 0]);  arg453_1 = None
        addmm_146: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg454_1, view_392, permute_196);  arg454_1 = view_392 = permute_196 = None
        view_393: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_146, [256, 128, 512]);  addmm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_38: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_393);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_394: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_38, [32768, 512]);  relu_38 = None
        permute_197: "f32[512, 128]" = torch.ops.aten.permute.default(arg455_1, [1, 0]);  arg455_1 = None
        addmm_147: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg456_1, view_394, permute_197);  arg456_1 = view_394 = permute_197 = None
        view_395: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_147, [256, 128, 128]);  addmm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_149: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_395, add_148);  view_395 = add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_98: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_149, arg457_1);  add_149 = arg457_1 = None
        add_150: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_98, arg458_1);  mul_98 = arg458_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_396: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_150, [32768, 128])
        permute_198: "f32[128, 512]" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
        addmm_148: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg460_1, view_396, permute_198);  arg460_1 = view_396 = permute_198 = None
        view_397: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_148, [256, 128, 512]);  addmm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_39: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_397);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_398: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_39, [32768, 512]);  relu_39 = None
        permute_199: "f32[512, 128]" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        addmm_149: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg462_1, view_398, permute_199);  arg462_1 = view_398 = permute_199 = None
        view_399: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_149, [256, 128, 128]);  addmm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_151: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_399, add_150);  view_399 = add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_99: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_151, arg463_1);  add_151 = arg463_1 = None
        add_152: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_99, arg464_1);  mul_99 = arg464_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_400: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_152, [32768, 128]);  add_152 = None
        permute_200: "f32[128, 512]" = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
        addmm_150: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg466_1, view_400, permute_200);  arg466_1 = view_400 = permute_200 = None
        view_401: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_150, [256, 128, 512]);  addmm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_153: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_401, add_139);  view_401 = add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_100: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_153, arg467_1);  add_153 = arg467_1 = None
        add_154: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_100, arg468_1);  mul_100 = arg468_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_404: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_154, [32768, 512])
        permute_202: "f32[512, 128]" = torch.ops.aten.permute.default(arg473_1, [1, 0]);  arg473_1 = None
        addmm_152: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg474_1, view_404, permute_202);  arg474_1 = view_404 = permute_202 = None
        view_405: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_152, [256, 128, 128]);  addmm_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_102: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_405, arg475_1);  view_405 = arg475_1 = None
        add_156: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_102, arg476_1);  mul_102 = arg476_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_406: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_156, [32768, 128])
        permute_203: "f32[128, 128]" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_153: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg478_1, view_406, permute_203);  arg478_1 = view_406 = permute_203 = None
        view_407: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_153, [256, 128, 128]);  addmm_153 = None
        view_408: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_407, [256, 128, -1, 32]);  view_407 = None
        permute_204: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_103: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_204, 0.4204482076268573);  permute_204 = None
        expand_41: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_103, [256, 4, 128, 32]);  mul_103 = None
        clone_51: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_415: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_51, [1024, 128, 32]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_409: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_156, [32768, 128]);  add_156 = None
        permute_205: "f32[128, 128]" = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
        addmm_154: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg480_1, view_409, permute_205);  arg480_1 = view_409 = permute_205 = None
        view_410: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_154, [256, 128, 128]);  addmm_154 = None
        view_411: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_410, [256, 128, -1, 32]);  view_410 = None
        permute_206: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_209: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2]);  permute_206 = None
        mul_104: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_209, 0.4204482076268573);  permute_209 = None
        expand_42: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_104, [256, 4, 32, 128]);  mul_104 = None
        clone_52: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_416: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_52, [1024, 32, 128]);  clone_52 = None
        bmm_20: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_415, view_416);  view_415 = view_416 = None
        view_417: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [256, 4, 128, 128]);  bmm_20 = None
        full_default_32: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_32, full_default_31);  full_default_32 = full_default_31 = None
        add_157: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_417, where_20);  view_417 = where_20 = None
        eq_10: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_157, -inf)
        logical_not_20: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_default_33: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_10: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_157, [-1], True)
        sub_10: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_157, amax_10);  add_157 = amax_10 = None
        exp_10: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_11: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        where_21: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_21, full_default_33, div_10);  logical_not_21 = full_default_33 = div_10 = None
        expand_43: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_21, [256, 4, 128, 128]);  where_21 = None
        view_418: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_43, [1024, 128, 128]);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_412: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_154, [32768, 512])
        permute_207: "f32[512, 128]" = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
        addmm_155: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg482_1, view_412, permute_207);  arg482_1 = view_412 = permute_207 = None
        view_413: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_155, [256, 128, 128]);  addmm_155 = None
        view_414: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_413, [256, 128, -1, 32]);  view_413 = None
        permute_208: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_44: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_208, [256, 4, 128, 32]);  permute_208 = None
        clone_53: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_419: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_53, [1024, 128, 32]);  clone_53 = None
        bmm_21: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_418, view_419);  view_418 = view_419 = None
        view_420: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_21, [256, 4, 128, 32]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_210: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        clone_54: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_421: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_54, [256, 128, -1]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_422: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_421, [32768, 128]);  view_421 = None
        permute_211: "f32[128, 128]" = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
        addmm_156: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg484_1, view_422, permute_211);  arg484_1 = view_422 = permute_211 = None
        view_423: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_156, [256, 128, 128]);  addmm_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_402: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_154, [32768, 512])
        permute_201: "f32[512, 128]" = torch.ops.aten.permute.default(arg469_1, [1, 0]);  arg469_1 = None
        addmm_151: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg470_1, view_402, permute_201);  arg470_1 = view_402 = permute_201 = None
        view_403: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_151, [256, 128, 128]);  addmm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_101: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_403, arg471_1);  view_403 = arg471_1 = None
        add_155: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_101, arg472_1);  mul_101 = arg472_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_158: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_423, add_155);  view_423 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_105: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_158, arg485_1);  add_158 = arg485_1 = None
        add_159: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_105, arg486_1);  mul_105 = arg486_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_424: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_159, [32768, 128])
        permute_212: "f32[128, 512]" = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
        addmm_157: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg488_1, view_424, permute_212);  arg488_1 = view_424 = permute_212 = None
        view_425: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_157, [256, 128, 512]);  addmm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_40: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_425);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_426: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_40, [32768, 512]);  relu_40 = None
        permute_213: "f32[512, 128]" = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
        addmm_158: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg490_1, view_426, permute_213);  arg490_1 = view_426 = permute_213 = None
        view_427: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_158, [256, 128, 128]);  addmm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_160: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_427, add_159);  view_427 = add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_106: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_160, arg491_1);  add_160 = arg491_1 = None
        add_161: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_106, arg492_1);  mul_106 = arg492_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_428: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_161, [32768, 128])
        permute_214: "f32[128, 512]" = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_159: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg494_1, view_428, permute_214);  arg494_1 = view_428 = permute_214 = None
        view_429: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_159, [256, 128, 512]);  addmm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_41: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_429);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_430: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_41, [32768, 512]);  relu_41 = None
        permute_215: "f32[512, 128]" = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
        addmm_160: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg496_1, view_430, permute_215);  arg496_1 = view_430 = permute_215 = None
        view_431: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_160, [256, 128, 128]);  addmm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_162: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_431, add_161);  view_431 = add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_107: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_162, arg497_1);  add_162 = arg497_1 = None
        add_163: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_107, arg498_1);  mul_107 = arg498_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_432: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_163, [32768, 128])
        permute_216: "f32[128, 512]" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        addmm_161: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg500_1, view_432, permute_216);  arg500_1 = view_432 = permute_216 = None
        view_433: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_161, [256, 128, 512]);  addmm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_42: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_433);  view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_434: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_42, [32768, 512]);  relu_42 = None
        permute_217: "f32[512, 128]" = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
        addmm_162: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg502_1, view_434, permute_217);  arg502_1 = view_434 = permute_217 = None
        view_435: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_162, [256, 128, 128]);  addmm_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_164: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_435, add_163);  view_435 = add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_108: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_164, arg503_1);  add_164 = arg503_1 = None
        add_165: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_108, arg504_1);  mul_108 = arg504_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_436: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_165, [32768, 128])
        permute_218: "f32[128, 512]" = torch.ops.aten.permute.default(arg505_1, [1, 0]);  arg505_1 = None
        addmm_163: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg506_1, view_436, permute_218);  arg506_1 = view_436 = permute_218 = None
        view_437: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_163, [256, 128, 512]);  addmm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_43: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_437);  view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_438: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_43, [32768, 512]);  relu_43 = None
        permute_219: "f32[512, 128]" = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
        addmm_164: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg508_1, view_438, permute_219);  arg508_1 = view_438 = permute_219 = None
        view_439: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_164, [256, 128, 128]);  addmm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_166: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_439, add_165);  view_439 = add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_109: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_166, arg509_1);  add_166 = arg509_1 = None
        add_167: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_109, arg510_1);  mul_109 = arg510_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_440: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_167, [32768, 128]);  add_167 = None
        permute_220: "f32[128, 512]" = torch.ops.aten.permute.default(arg511_1, [1, 0]);  arg511_1 = None
        addmm_165: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg512_1, view_440, permute_220);  arg512_1 = view_440 = permute_220 = None
        view_441: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_165, [256, 128, 512]);  addmm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_168: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_441, add_154);  view_441 = add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_110: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_168, arg513_1);  add_168 = arg513_1 = None
        add_169: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_110, arg514_1);  mul_110 = arg514_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_444: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_169, [32768, 512])
        permute_222: "f32[512, 128]" = torch.ops.aten.permute.default(arg519_1, [1, 0]);  arg519_1 = None
        addmm_167: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg520_1, view_444, permute_222);  arg520_1 = view_444 = permute_222 = None
        view_445: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_167, [256, 128, 128]);  addmm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_112: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_445, arg521_1);  view_445 = arg521_1 = None
        add_171: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_112, arg522_1);  mul_112 = arg522_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_446: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_171, [32768, 128])
        permute_223: "f32[128, 128]" = torch.ops.aten.permute.default(arg523_1, [1, 0]);  arg523_1 = None
        addmm_168: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg524_1, view_446, permute_223);  arg524_1 = view_446 = permute_223 = None
        view_447: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_168, [256, 128, 128]);  addmm_168 = None
        view_448: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_447, [256, 128, -1, 32]);  view_447 = None
        permute_224: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_113: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_224, 0.4204482076268573);  permute_224 = None
        expand_45: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_113, [256, 4, 128, 32]);  mul_113 = None
        clone_56: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_455: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_56, [1024, 128, 32]);  clone_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_449: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_171, [32768, 128]);  add_171 = None
        permute_225: "f32[128, 128]" = torch.ops.aten.permute.default(arg525_1, [1, 0]);  arg525_1 = None
        addmm_169: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg526_1, view_449, permute_225);  arg526_1 = view_449 = permute_225 = None
        view_450: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_169, [256, 128, 128]);  addmm_169 = None
        view_451: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_450, [256, 128, -1, 32]);  view_450 = None
        permute_226: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_229: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_226, [0, 1, 3, 2]);  permute_226 = None
        mul_114: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_229, 0.4204482076268573);  permute_229 = None
        expand_46: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_114, [256, 4, 32, 128]);  mul_114 = None
        clone_57: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_456: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_57, [1024, 32, 128]);  clone_57 = None
        bmm_22: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_455, view_456);  view_455 = view_456 = None
        view_457: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [256, 4, 128, 128]);  bmm_22 = None
        full_default_35: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_35, full_default_34);  full_default_35 = full_default_34 = None
        add_172: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_457, where_22);  view_457 = where_22 = None
        eq_11: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_172, -inf)
        logical_not_22: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_default_36: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_11: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_172, [-1], True)
        sub_11: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_172, amax_11);  add_172 = amax_11 = None
        exp_11: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_12: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        where_23: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_23, full_default_36, div_11);  logical_not_23 = full_default_36 = div_11 = None
        expand_47: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_23, [256, 4, 128, 128]);  where_23 = None
        view_458: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_47, [1024, 128, 128]);  expand_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_452: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_169, [32768, 512])
        permute_227: "f32[512, 128]" = torch.ops.aten.permute.default(arg527_1, [1, 0]);  arg527_1 = None
        addmm_170: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg528_1, view_452, permute_227);  arg528_1 = view_452 = permute_227 = None
        view_453: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_170, [256, 128, 128]);  addmm_170 = None
        view_454: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_453, [256, 128, -1, 32]);  view_453 = None
        permute_228: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_48: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_228, [256, 4, 128, 32]);  permute_228 = None
        clone_58: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_459: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_58, [1024, 128, 32]);  clone_58 = None
        bmm_23: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_458, view_459);  view_458 = view_459 = None
        view_460: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_23, [256, 4, 128, 32]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_230: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        clone_59: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_461: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_59, [256, 128, -1]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_462: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_461, [32768, 128]);  view_461 = None
        permute_231: "f32[128, 128]" = torch.ops.aten.permute.default(arg529_1, [1, 0]);  arg529_1 = None
        addmm_171: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg530_1, view_462, permute_231);  arg530_1 = view_462 = permute_231 = None
        view_463: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_171, [256, 128, 128]);  addmm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_442: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_169, [32768, 512])
        permute_221: "f32[512, 128]" = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
        addmm_166: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg516_1, view_442, permute_221);  arg516_1 = view_442 = permute_221 = None
        view_443: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_166, [256, 128, 128]);  addmm_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_111: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_443, arg517_1);  view_443 = arg517_1 = None
        add_170: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_111, arg518_1);  mul_111 = arg518_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_173: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_463, add_170);  view_463 = add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_115: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_173, arg531_1);  add_173 = arg531_1 = None
        add_174: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_115, arg532_1);  mul_115 = arg532_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_464: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_174, [32768, 128])
        permute_232: "f32[128, 512]" = torch.ops.aten.permute.default(arg533_1, [1, 0]);  arg533_1 = None
        addmm_172: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg534_1, view_464, permute_232);  arg534_1 = view_464 = permute_232 = None
        view_465: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_172, [256, 128, 512]);  addmm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_44: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_465);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_466: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_44, [32768, 512]);  relu_44 = None
        permute_233: "f32[512, 128]" = torch.ops.aten.permute.default(arg535_1, [1, 0]);  arg535_1 = None
        addmm_173: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg536_1, view_466, permute_233);  arg536_1 = view_466 = permute_233 = None
        view_467: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_173, [256, 128, 128]);  addmm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_175: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_467, add_174);  view_467 = add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_116: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_175, arg537_1);  add_175 = arg537_1 = None
        add_176: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_116, arg538_1);  mul_116 = arg538_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_468: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_176, [32768, 128])
        permute_234: "f32[128, 512]" = torch.ops.aten.permute.default(arg539_1, [1, 0]);  arg539_1 = None
        addmm_174: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg540_1, view_468, permute_234);  arg540_1 = view_468 = permute_234 = None
        view_469: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_174, [256, 128, 512]);  addmm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_45: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_469);  view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_470: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_45, [32768, 512]);  relu_45 = None
        permute_235: "f32[512, 128]" = torch.ops.aten.permute.default(arg541_1, [1, 0]);  arg541_1 = None
        addmm_175: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg542_1, view_470, permute_235);  arg542_1 = view_470 = permute_235 = None
        view_471: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_175, [256, 128, 128]);  addmm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_177: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_471, add_176);  view_471 = add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_117: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_177, arg543_1);  add_177 = arg543_1 = None
        add_178: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_117, arg544_1);  mul_117 = arg544_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_472: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_178, [32768, 128])
        permute_236: "f32[128, 512]" = torch.ops.aten.permute.default(arg545_1, [1, 0]);  arg545_1 = None
        addmm_176: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg546_1, view_472, permute_236);  arg546_1 = view_472 = permute_236 = None
        view_473: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_176, [256, 128, 512]);  addmm_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_46: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_473);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_474: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_46, [32768, 512]);  relu_46 = None
        permute_237: "f32[512, 128]" = torch.ops.aten.permute.default(arg547_1, [1, 0]);  arg547_1 = None
        addmm_177: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg548_1, view_474, permute_237);  arg548_1 = view_474 = permute_237 = None
        view_475: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_177, [256, 128, 128]);  addmm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_179: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_475, add_178);  view_475 = add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_118: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_179, arg549_1);  add_179 = arg549_1 = None
        add_180: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_118, arg550_1);  mul_118 = arg550_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_476: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_180, [32768, 128])
        permute_238: "f32[128, 512]" = torch.ops.aten.permute.default(arg551_1, [1, 0]);  arg551_1 = None
        addmm_178: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg552_1, view_476, permute_238);  arg552_1 = view_476 = permute_238 = None
        view_477: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_178, [256, 128, 512]);  addmm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_47: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_477);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_478: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_47, [32768, 512]);  relu_47 = None
        permute_239: "f32[512, 128]" = torch.ops.aten.permute.default(arg553_1, [1, 0]);  arg553_1 = None
        addmm_179: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg554_1, view_478, permute_239);  arg554_1 = view_478 = permute_239 = None
        view_479: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_179, [256, 128, 128]);  addmm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_181: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_479, add_180);  view_479 = add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_119: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_181, arg555_1);  add_181 = arg555_1 = None
        add_182: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_119, arg556_1);  mul_119 = arg556_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_480: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_182, [32768, 128]);  add_182 = None
        permute_240: "f32[128, 512]" = torch.ops.aten.permute.default(arg557_1, [1, 0]);  arg557_1 = None
        addmm_180: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg558_1, view_480, permute_240);  arg558_1 = view_480 = permute_240 = None
        view_481: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_180, [256, 128, 512]);  addmm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_183: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_481, add_169);  view_481 = add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_120: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_183, arg559_1);  add_183 = arg559_1 = None
        add_184: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_120, arg560_1);  mul_120 = arg560_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_484: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_184, [32768, 512])
        permute_242: "f32[512, 128]" = torch.ops.aten.permute.default(arg565_1, [1, 0]);  arg565_1 = None
        addmm_182: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg566_1, view_484, permute_242);  arg566_1 = view_484 = permute_242 = None
        view_485: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_182, [256, 128, 128]);  addmm_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_122: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_485, arg567_1);  view_485 = arg567_1 = None
        add_186: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_122, arg568_1);  mul_122 = arg568_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_486: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_186, [32768, 128])
        permute_243: "f32[128, 128]" = torch.ops.aten.permute.default(arg569_1, [1, 0]);  arg569_1 = None
        addmm_183: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg570_1, view_486, permute_243);  arg570_1 = view_486 = permute_243 = None
        view_487: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_183, [256, 128, 128]);  addmm_183 = None
        view_488: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_487, [256, 128, -1, 32]);  view_487 = None
        permute_244: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_123: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_244, 0.4204482076268573);  permute_244 = None
        expand_49: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_123, [256, 4, 128, 32]);  mul_123 = None
        clone_61: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_495: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_61, [1024, 128, 32]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_489: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_186, [32768, 128]);  add_186 = None
        permute_245: "f32[128, 128]" = torch.ops.aten.permute.default(arg571_1, [1, 0]);  arg571_1 = None
        addmm_184: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg572_1, view_489, permute_245);  arg572_1 = view_489 = permute_245 = None
        view_490: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_184, [256, 128, 128]);  addmm_184 = None
        view_491: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_490, [256, 128, -1, 32]);  view_490 = None
        permute_246: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_249: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2]);  permute_246 = None
        mul_124: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_249, 0.4204482076268573);  permute_249 = None
        expand_50: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_124, [256, 4, 32, 128]);  mul_124 = None
        clone_62: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_496: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_62, [1024, 32, 128]);  clone_62 = None
        bmm_24: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_495, view_496);  view_495 = view_496 = None
        view_497: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [256, 4, 128, 128]);  bmm_24 = None
        full_default_38: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_38, full_default_37);  full_default_38 = full_default_37 = None
        add_187: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_497, where_24);  view_497 = where_24 = None
        eq_12: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_187, -inf)
        logical_not_24: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_12);  eq_12 = None
        any_13: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_24, -1, True);  logical_not_24 = None
        logical_not_25: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        full_default_39: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_12: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_187, [-1], True)
        sub_12: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_187, amax_12);  add_187 = amax_12 = None
        exp_12: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_13: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        where_25: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_25, full_default_39, div_12);  logical_not_25 = full_default_39 = div_12 = None
        expand_51: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_25, [256, 4, 128, 128]);  where_25 = None
        view_498: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_51, [1024, 128, 128]);  expand_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_492: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_184, [32768, 512])
        permute_247: "f32[512, 128]" = torch.ops.aten.permute.default(arg573_1, [1, 0]);  arg573_1 = None
        addmm_185: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg574_1, view_492, permute_247);  arg574_1 = view_492 = permute_247 = None
        view_493: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_185, [256, 128, 128]);  addmm_185 = None
        view_494: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_493, [256, 128, -1, 32]);  view_493 = None
        permute_248: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_52: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_248, [256, 4, 128, 32]);  permute_248 = None
        clone_63: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_499: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_63, [1024, 128, 32]);  clone_63 = None
        bmm_25: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_498, view_499);  view_498 = view_499 = None
        view_500: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_25, [256, 4, 128, 32]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_250: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
        clone_64: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_501: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_64, [256, 128, -1]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_502: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_501, [32768, 128]);  view_501 = None
        permute_251: "f32[128, 128]" = torch.ops.aten.permute.default(arg575_1, [1, 0]);  arg575_1 = None
        addmm_186: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg576_1, view_502, permute_251);  arg576_1 = view_502 = permute_251 = None
        view_503: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_186, [256, 128, 128]);  addmm_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_482: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_184, [32768, 512])
        permute_241: "f32[512, 128]" = torch.ops.aten.permute.default(arg561_1, [1, 0]);  arg561_1 = None
        addmm_181: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg562_1, view_482, permute_241);  arg562_1 = view_482 = permute_241 = None
        view_483: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_181, [256, 128, 128]);  addmm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_121: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_483, arg563_1);  view_483 = arg563_1 = None
        add_185: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_121, arg564_1);  mul_121 = arg564_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_188: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_503, add_185);  view_503 = add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_125: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_188, arg577_1);  add_188 = arg577_1 = None
        add_189: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_125, arg578_1);  mul_125 = arg578_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_189, [32768, 128])
        permute_252: "f32[128, 512]" = torch.ops.aten.permute.default(arg579_1, [1, 0]);  arg579_1 = None
        addmm_187: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg580_1, view_504, permute_252);  arg580_1 = view_504 = permute_252 = None
        view_505: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_187, [256, 128, 512]);  addmm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_48: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_505);  view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_506: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_48, [32768, 512]);  relu_48 = None
        permute_253: "f32[512, 128]" = torch.ops.aten.permute.default(arg581_1, [1, 0]);  arg581_1 = None
        addmm_188: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg582_1, view_506, permute_253);  arg582_1 = view_506 = permute_253 = None
        view_507: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_188, [256, 128, 128]);  addmm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_190: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_507, add_189);  view_507 = add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_126: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_190, arg583_1);  add_190 = arg583_1 = None
        add_191: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_126, arg584_1);  mul_126 = arg584_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_508: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_191, [32768, 128])
        permute_254: "f32[128, 512]" = torch.ops.aten.permute.default(arg585_1, [1, 0]);  arg585_1 = None
        addmm_189: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg586_1, view_508, permute_254);  arg586_1 = view_508 = permute_254 = None
        view_509: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_189, [256, 128, 512]);  addmm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_49: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_509);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_510: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_49, [32768, 512]);  relu_49 = None
        permute_255: "f32[512, 128]" = torch.ops.aten.permute.default(arg587_1, [1, 0]);  arg587_1 = None
        addmm_190: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg588_1, view_510, permute_255);  arg588_1 = view_510 = permute_255 = None
        view_511: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_190, [256, 128, 128]);  addmm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_192: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_511, add_191);  view_511 = add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_127: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_192, arg589_1);  add_192 = arg589_1 = None
        add_193: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_127, arg590_1);  mul_127 = arg590_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_512: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_193, [32768, 128])
        permute_256: "f32[128, 512]" = torch.ops.aten.permute.default(arg591_1, [1, 0]);  arg591_1 = None
        addmm_191: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg592_1, view_512, permute_256);  arg592_1 = view_512 = permute_256 = None
        view_513: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_191, [256, 128, 512]);  addmm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_50: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_513);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_514: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_50, [32768, 512]);  relu_50 = None
        permute_257: "f32[512, 128]" = torch.ops.aten.permute.default(arg593_1, [1, 0]);  arg593_1 = None
        addmm_192: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg594_1, view_514, permute_257);  arg594_1 = view_514 = permute_257 = None
        view_515: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_192, [256, 128, 128]);  addmm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_194: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_515, add_193);  view_515 = add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_128: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_194, arg595_1);  add_194 = arg595_1 = None
        add_195: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_128, arg596_1);  mul_128 = arg596_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_516: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_195, [32768, 128])
        permute_258: "f32[128, 512]" = torch.ops.aten.permute.default(arg597_1, [1, 0]);  arg597_1 = None
        addmm_193: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg598_1, view_516, permute_258);  arg598_1 = view_516 = permute_258 = None
        view_517: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_193, [256, 128, 512]);  addmm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_51: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_517);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_518: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_51, [32768, 512]);  relu_51 = None
        permute_259: "f32[512, 128]" = torch.ops.aten.permute.default(arg599_1, [1, 0]);  arg599_1 = None
        addmm_194: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg600_1, view_518, permute_259);  arg600_1 = view_518 = permute_259 = None
        view_519: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_194, [256, 128, 128]);  addmm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_196: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_519, add_195);  view_519 = add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_129: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_196, arg601_1);  add_196 = arg601_1 = None
        add_197: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_129, arg602_1);  mul_129 = arg602_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_520: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_197, [32768, 128]);  add_197 = None
        permute_260: "f32[128, 512]" = torch.ops.aten.permute.default(arg603_1, [1, 0]);  arg603_1 = None
        addmm_195: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg604_1, view_520, permute_260);  arg604_1 = view_520 = permute_260 = None
        view_521: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_195, [256, 128, 512]);  addmm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_198: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_521, add_184);  view_521 = add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_130: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_198, arg605_1);  add_198 = arg605_1 = None
        add_199: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_130, arg606_1);  mul_130 = arg606_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_524: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_199, [32768, 512])
        permute_262: "f32[512, 128]" = torch.ops.aten.permute.default(arg611_1, [1, 0]);  arg611_1 = None
        addmm_197: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg612_1, view_524, permute_262);  arg612_1 = view_524 = permute_262 = None
        view_525: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_197, [256, 128, 128]);  addmm_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_132: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_525, arg613_1);  view_525 = arg613_1 = None
        add_201: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_132, arg614_1);  mul_132 = arg614_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_526: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_201, [32768, 128])
        permute_263: "f32[128, 128]" = torch.ops.aten.permute.default(arg615_1, [1, 0]);  arg615_1 = None
        addmm_198: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg616_1, view_526, permute_263);  arg616_1 = view_526 = permute_263 = None
        view_527: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_198, [256, 128, 128]);  addmm_198 = None
        view_528: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_527, [256, 128, -1, 32]);  view_527 = None
        permute_264: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_133: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_264, 0.4204482076268573);  permute_264 = None
        expand_53: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_133, [256, 4, 128, 32]);  mul_133 = None
        clone_66: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_535: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_66, [1024, 128, 32]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_529: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_201, [32768, 128]);  add_201 = None
        permute_265: "f32[128, 128]" = torch.ops.aten.permute.default(arg617_1, [1, 0]);  arg617_1 = None
        addmm_199: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg618_1, view_529, permute_265);  arg618_1 = view_529 = permute_265 = None
        view_530: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_199, [256, 128, 128]);  addmm_199 = None
        view_531: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_530, [256, 128, -1, 32]);  view_530 = None
        permute_266: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_269: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_266, [0, 1, 3, 2]);  permute_266 = None
        mul_134: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_269, 0.4204482076268573);  permute_269 = None
        expand_54: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_134, [256, 4, 32, 128]);  mul_134 = None
        clone_67: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_536: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_67, [1024, 32, 128]);  clone_67 = None
        bmm_26: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_535, view_536);  view_535 = view_536 = None
        view_537: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [256, 4, 128, 128]);  bmm_26 = None
        full_default_41: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_41, full_default_40);  full_default_41 = full_default_40 = None
        add_202: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_537, where_26);  view_537 = where_26 = None
        eq_13: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_202, -inf)
        logical_not_26: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_13);  eq_13 = None
        any_14: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_26, -1, True);  logical_not_26 = None
        logical_not_27: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        full_default_42: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_13: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_202, [-1], True)
        sub_13: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_202, amax_13);  add_202 = amax_13 = None
        exp_13: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_14: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        where_27: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_27, full_default_42, div_13);  logical_not_27 = full_default_42 = div_13 = None
        expand_55: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_27, [256, 4, 128, 128]);  where_27 = None
        view_538: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_55, [1024, 128, 128]);  expand_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_532: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_199, [32768, 512])
        permute_267: "f32[512, 128]" = torch.ops.aten.permute.default(arg619_1, [1, 0]);  arg619_1 = None
        addmm_200: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg620_1, view_532, permute_267);  arg620_1 = view_532 = permute_267 = None
        view_533: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_200, [256, 128, 128]);  addmm_200 = None
        view_534: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_533, [256, 128, -1, 32]);  view_533 = None
        permute_268: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_56: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_268, [256, 4, 128, 32]);  permute_268 = None
        clone_68: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_539: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_68, [1024, 128, 32]);  clone_68 = None
        bmm_27: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_538, view_539);  view_538 = view_539 = None
        view_540: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_27, [256, 4, 128, 32]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_270: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
        clone_69: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_541: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_69, [256, 128, -1]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_542: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_541, [32768, 128]);  view_541 = None
        permute_271: "f32[128, 128]" = torch.ops.aten.permute.default(arg621_1, [1, 0]);  arg621_1 = None
        addmm_201: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg622_1, view_542, permute_271);  arg622_1 = view_542 = permute_271 = None
        view_543: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_201, [256, 128, 128]);  addmm_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_522: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_199, [32768, 512])
        permute_261: "f32[512, 128]" = torch.ops.aten.permute.default(arg607_1, [1, 0]);  arg607_1 = None
        addmm_196: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg608_1, view_522, permute_261);  arg608_1 = view_522 = permute_261 = None
        view_523: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_196, [256, 128, 128]);  addmm_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_131: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_523, arg609_1);  view_523 = arg609_1 = None
        add_200: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_131, arg610_1);  mul_131 = arg610_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_203: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_543, add_200);  view_543 = add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_135: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_203, arg623_1);  add_203 = arg623_1 = None
        add_204: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_135, arg624_1);  mul_135 = arg624_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_544: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_204, [32768, 128])
        permute_272: "f32[128, 512]" = torch.ops.aten.permute.default(arg625_1, [1, 0]);  arg625_1 = None
        addmm_202: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg626_1, view_544, permute_272);  arg626_1 = view_544 = permute_272 = None
        view_545: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_202, [256, 128, 512]);  addmm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_52: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_545);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_546: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_52, [32768, 512]);  relu_52 = None
        permute_273: "f32[512, 128]" = torch.ops.aten.permute.default(arg627_1, [1, 0]);  arg627_1 = None
        addmm_203: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg628_1, view_546, permute_273);  arg628_1 = view_546 = permute_273 = None
        view_547: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_203, [256, 128, 128]);  addmm_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_205: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_547, add_204);  view_547 = add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_136: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_205, arg629_1);  add_205 = arg629_1 = None
        add_206: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_136, arg630_1);  mul_136 = arg630_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_548: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_206, [32768, 128])
        permute_274: "f32[128, 512]" = torch.ops.aten.permute.default(arg631_1, [1, 0]);  arg631_1 = None
        addmm_204: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg632_1, view_548, permute_274);  arg632_1 = view_548 = permute_274 = None
        view_549: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_204, [256, 128, 512]);  addmm_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_53: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_549);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_550: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_53, [32768, 512]);  relu_53 = None
        permute_275: "f32[512, 128]" = torch.ops.aten.permute.default(arg633_1, [1, 0]);  arg633_1 = None
        addmm_205: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg634_1, view_550, permute_275);  arg634_1 = view_550 = permute_275 = None
        view_551: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_205, [256, 128, 128]);  addmm_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_207: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_551, add_206);  view_551 = add_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_137: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_207, arg635_1);  add_207 = arg635_1 = None
        add_208: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_137, arg636_1);  mul_137 = arg636_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_552: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_208, [32768, 128])
        permute_276: "f32[128, 512]" = torch.ops.aten.permute.default(arg637_1, [1, 0]);  arg637_1 = None
        addmm_206: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg638_1, view_552, permute_276);  arg638_1 = view_552 = permute_276 = None
        view_553: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_206, [256, 128, 512]);  addmm_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_54: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_553);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_554: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_54, [32768, 512]);  relu_54 = None
        permute_277: "f32[512, 128]" = torch.ops.aten.permute.default(arg639_1, [1, 0]);  arg639_1 = None
        addmm_207: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg640_1, view_554, permute_277);  arg640_1 = view_554 = permute_277 = None
        view_555: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_207, [256, 128, 128]);  addmm_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_209: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_555, add_208);  view_555 = add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_138: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_209, arg641_1);  add_209 = arg641_1 = None
        add_210: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_138, arg642_1);  mul_138 = arg642_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_556: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_210, [32768, 128])
        permute_278: "f32[128, 512]" = torch.ops.aten.permute.default(arg643_1, [1, 0]);  arg643_1 = None
        addmm_208: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg644_1, view_556, permute_278);  arg644_1 = view_556 = permute_278 = None
        view_557: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_208, [256, 128, 512]);  addmm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_55: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_557);  view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_558: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_55, [32768, 512]);  relu_55 = None
        permute_279: "f32[512, 128]" = torch.ops.aten.permute.default(arg645_1, [1, 0]);  arg645_1 = None
        addmm_209: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg646_1, view_558, permute_279);  arg646_1 = view_558 = permute_279 = None
        view_559: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_209, [256, 128, 128]);  addmm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_211: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_559, add_210);  view_559 = add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_139: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_211, arg647_1);  add_211 = arg647_1 = None
        add_212: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_139, arg648_1);  mul_139 = arg648_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_560: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_212, [32768, 128]);  add_212 = None
        permute_280: "f32[128, 512]" = torch.ops.aten.permute.default(arg649_1, [1, 0]);  arg649_1 = None
        addmm_210: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg650_1, view_560, permute_280);  arg650_1 = view_560 = permute_280 = None
        view_561: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_210, [256, 128, 512]);  addmm_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_213: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_561, add_199);  view_561 = add_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_140: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_213, arg651_1);  add_213 = arg651_1 = None
        add_214: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_140, arg652_1);  mul_140 = arg652_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_564: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_214, [32768, 512])
        permute_282: "f32[512, 128]" = torch.ops.aten.permute.default(arg657_1, [1, 0]);  arg657_1 = None
        addmm_212: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg658_1, view_564, permute_282);  arg658_1 = view_564 = permute_282 = None
        view_565: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_212, [256, 128, 128]);  addmm_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_142: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_565, arg659_1);  view_565 = arg659_1 = None
        add_216: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_142, arg660_1);  mul_142 = arg660_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_566: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_216, [32768, 128])
        permute_283: "f32[128, 128]" = torch.ops.aten.permute.default(arg661_1, [1, 0]);  arg661_1 = None
        addmm_213: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg662_1, view_566, permute_283);  arg662_1 = view_566 = permute_283 = None
        view_567: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_213, [256, 128, 128]);  addmm_213 = None
        view_568: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_567, [256, 128, -1, 32]);  view_567 = None
        permute_284: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_143: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_284, 0.4204482076268573);  permute_284 = None
        expand_57: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_143, [256, 4, 128, 32]);  mul_143 = None
        clone_71: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_575: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_71, [1024, 128, 32]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_569: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_216, [32768, 128]);  add_216 = None
        permute_285: "f32[128, 128]" = torch.ops.aten.permute.default(arg663_1, [1, 0]);  arg663_1 = None
        addmm_214: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg664_1, view_569, permute_285);  arg664_1 = view_569 = permute_285 = None
        view_570: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_214, [256, 128, 128]);  addmm_214 = None
        view_571: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_570, [256, 128, -1, 32]);  view_570 = None
        permute_286: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_289: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_286, [0, 1, 3, 2]);  permute_286 = None
        mul_144: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_289, 0.4204482076268573);  permute_289 = None
        expand_58: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_144, [256, 4, 32, 128]);  mul_144 = None
        clone_72: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_576: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_72, [1024, 32, 128]);  clone_72 = None
        bmm_28: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_575, view_576);  view_575 = view_576 = None
        view_577: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [256, 4, 128, 128]);  bmm_28 = None
        full_default_44: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_44, full_default_43);  full_default_44 = full_default_43 = None
        add_217: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_577, where_28);  view_577 = where_28 = None
        eq_14: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_217, -inf)
        logical_not_28: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_14);  eq_14 = None
        any_15: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_28, -1, True);  logical_not_28 = None
        logical_not_29: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        full_default_45: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_14: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_217, [-1], True)
        sub_14: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_217, amax_14);  add_217 = amax_14 = None
        exp_14: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_15: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        where_29: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_29, full_default_45, div_14);  logical_not_29 = full_default_45 = div_14 = None
        expand_59: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_29, [256, 4, 128, 128]);  where_29 = None
        view_578: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_59, [1024, 128, 128]);  expand_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_572: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_214, [32768, 512])
        permute_287: "f32[512, 128]" = torch.ops.aten.permute.default(arg665_1, [1, 0]);  arg665_1 = None
        addmm_215: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg666_1, view_572, permute_287);  arg666_1 = view_572 = permute_287 = None
        view_573: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_215, [256, 128, 128]);  addmm_215 = None
        view_574: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_573, [256, 128, -1, 32]);  view_573 = None
        permute_288: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_60: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_288, [256, 4, 128, 32]);  permute_288 = None
        clone_73: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_579: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_73, [1024, 128, 32]);  clone_73 = None
        bmm_29: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_578, view_579);  view_578 = view_579 = None
        view_580: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_29, [256, 4, 128, 32]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_290: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
        clone_74: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_581: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_74, [256, 128, -1]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_582: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_581, [32768, 128]);  view_581 = None
        permute_291: "f32[128, 128]" = torch.ops.aten.permute.default(arg667_1, [1, 0]);  arg667_1 = None
        addmm_216: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg668_1, view_582, permute_291);  arg668_1 = view_582 = permute_291 = None
        view_583: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_216, [256, 128, 128]);  addmm_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_562: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_214, [32768, 512])
        permute_281: "f32[512, 128]" = torch.ops.aten.permute.default(arg653_1, [1, 0]);  arg653_1 = None
        addmm_211: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg654_1, view_562, permute_281);  arg654_1 = view_562 = permute_281 = None
        view_563: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_211, [256, 128, 128]);  addmm_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_141: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_563, arg655_1);  view_563 = arg655_1 = None
        add_215: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_141, arg656_1);  mul_141 = arg656_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_218: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_583, add_215);  view_583 = add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_145: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_218, arg669_1);  add_218 = arg669_1 = None
        add_219: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_145, arg670_1);  mul_145 = arg670_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_584: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_219, [32768, 128])
        permute_292: "f32[128, 512]" = torch.ops.aten.permute.default(arg671_1, [1, 0]);  arg671_1 = None
        addmm_217: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg672_1, view_584, permute_292);  arg672_1 = view_584 = permute_292 = None
        view_585: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_217, [256, 128, 512]);  addmm_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_56: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_585);  view_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_586: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_56, [32768, 512]);  relu_56 = None
        permute_293: "f32[512, 128]" = torch.ops.aten.permute.default(arg673_1, [1, 0]);  arg673_1 = None
        addmm_218: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg674_1, view_586, permute_293);  arg674_1 = view_586 = permute_293 = None
        view_587: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_218, [256, 128, 128]);  addmm_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_220: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_587, add_219);  view_587 = add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_146: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_220, arg675_1);  add_220 = arg675_1 = None
        add_221: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_146, arg676_1);  mul_146 = arg676_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_588: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_221, [32768, 128])
        permute_294: "f32[128, 512]" = torch.ops.aten.permute.default(arg677_1, [1, 0]);  arg677_1 = None
        addmm_219: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg678_1, view_588, permute_294);  arg678_1 = view_588 = permute_294 = None
        view_589: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_219, [256, 128, 512]);  addmm_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_57: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_589);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_590: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_57, [32768, 512]);  relu_57 = None
        permute_295: "f32[512, 128]" = torch.ops.aten.permute.default(arg679_1, [1, 0]);  arg679_1 = None
        addmm_220: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg680_1, view_590, permute_295);  arg680_1 = view_590 = permute_295 = None
        view_591: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_220, [256, 128, 128]);  addmm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_222: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_591, add_221);  view_591 = add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_147: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_222, arg681_1);  add_222 = arg681_1 = None
        add_223: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_147, arg682_1);  mul_147 = arg682_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_592: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_223, [32768, 128])
        permute_296: "f32[128, 512]" = torch.ops.aten.permute.default(arg683_1, [1, 0]);  arg683_1 = None
        addmm_221: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg684_1, view_592, permute_296);  arg684_1 = view_592 = permute_296 = None
        view_593: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_221, [256, 128, 512]);  addmm_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_58: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_593);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_594: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_58, [32768, 512]);  relu_58 = None
        permute_297: "f32[512, 128]" = torch.ops.aten.permute.default(arg685_1, [1, 0]);  arg685_1 = None
        addmm_222: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg686_1, view_594, permute_297);  arg686_1 = view_594 = permute_297 = None
        view_595: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_222, [256, 128, 128]);  addmm_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_224: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_595, add_223);  view_595 = add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_148: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_224, arg687_1);  add_224 = arg687_1 = None
        add_225: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_148, arg688_1);  mul_148 = arg688_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_596: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_225, [32768, 128])
        permute_298: "f32[128, 512]" = torch.ops.aten.permute.default(arg689_1, [1, 0]);  arg689_1 = None
        addmm_223: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg690_1, view_596, permute_298);  arg690_1 = view_596 = permute_298 = None
        view_597: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_223, [256, 128, 512]);  addmm_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_59: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_597);  view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_598: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_59, [32768, 512]);  relu_59 = None
        permute_299: "f32[512, 128]" = torch.ops.aten.permute.default(arg691_1, [1, 0]);  arg691_1 = None
        addmm_224: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg692_1, view_598, permute_299);  arg692_1 = view_598 = permute_299 = None
        view_599: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_224, [256, 128, 128]);  addmm_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_226: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_599, add_225);  view_599 = add_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_149: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_226, arg693_1);  add_226 = arg693_1 = None
        add_227: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_149, arg694_1);  mul_149 = arg694_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_600: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_227, [32768, 128]);  add_227 = None
        permute_300: "f32[128, 512]" = torch.ops.aten.permute.default(arg695_1, [1, 0]);  arg695_1 = None
        addmm_225: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg696_1, view_600, permute_300);  arg696_1 = view_600 = permute_300 = None
        view_601: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_225, [256, 128, 512]);  addmm_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_228: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_601, add_214);  view_601 = add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_150: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_228, arg697_1);  add_228 = arg697_1 = None
        add_229: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_150, arg698_1);  mul_150 = arg698_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_604: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_229, [32768, 512])
        permute_302: "f32[512, 128]" = torch.ops.aten.permute.default(arg703_1, [1, 0]);  arg703_1 = None
        addmm_227: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg704_1, view_604, permute_302);  arg704_1 = view_604 = permute_302 = None
        view_605: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_227, [256, 128, 128]);  addmm_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_152: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_605, arg705_1);  view_605 = arg705_1 = None
        add_231: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_152, arg706_1);  mul_152 = arg706_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_606: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_231, [32768, 128])
        permute_303: "f32[128, 128]" = torch.ops.aten.permute.default(arg707_1, [1, 0]);  arg707_1 = None
        addmm_228: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg708_1, view_606, permute_303);  arg708_1 = view_606 = permute_303 = None
        view_607: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_228, [256, 128, 128]);  addmm_228 = None
        view_608: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_607, [256, 128, -1, 32]);  view_607 = None
        permute_304: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_153: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_304, 0.4204482076268573);  permute_304 = None
        expand_61: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_153, [256, 4, 128, 32]);  mul_153 = None
        clone_76: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_615: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_76, [1024, 128, 32]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_609: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_231, [32768, 128]);  add_231 = None
        permute_305: "f32[128, 128]" = torch.ops.aten.permute.default(arg709_1, [1, 0]);  arg709_1 = None
        addmm_229: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg710_1, view_609, permute_305);  arg710_1 = view_609 = permute_305 = None
        view_610: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_229, [256, 128, 128]);  addmm_229 = None
        view_611: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_610, [256, 128, -1, 32]);  view_610 = None
        permute_306: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_611, [0, 2, 1, 3]);  view_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_309: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_306, [0, 1, 3, 2]);  permute_306 = None
        mul_154: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_309, 0.4204482076268573);  permute_309 = None
        expand_62: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_154, [256, 4, 32, 128]);  mul_154 = None
        clone_77: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_616: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_77, [1024, 32, 128]);  clone_77 = None
        bmm_30: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_615, view_616);  view_615 = view_616 = None
        view_617: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [256, 4, 128, 128]);  bmm_30 = None
        full_default_47: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_47, full_default_46);  full_default_47 = full_default_46 = None
        add_232: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_617, where_30);  view_617 = where_30 = None
        eq_15: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_232, -inf)
        logical_not_30: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_15);  eq_15 = None
        any_16: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_30, -1, True);  logical_not_30 = None
        logical_not_31: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        full_default_48: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_15: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_232, [-1], True)
        sub_15: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_232, amax_15);  add_232 = amax_15 = None
        exp_15: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_16: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        where_31: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_31, full_default_48, div_15);  logical_not_31 = full_default_48 = div_15 = None
        expand_63: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_31, [256, 4, 128, 128]);  where_31 = None
        view_618: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_63, [1024, 128, 128]);  expand_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_612: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_229, [32768, 512])
        permute_307: "f32[512, 128]" = torch.ops.aten.permute.default(arg711_1, [1, 0]);  arg711_1 = None
        addmm_230: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg712_1, view_612, permute_307);  arg712_1 = view_612 = permute_307 = None
        view_613: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_230, [256, 128, 128]);  addmm_230 = None
        view_614: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_613, [256, 128, -1, 32]);  view_613 = None
        permute_308: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_64: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_308, [256, 4, 128, 32]);  permute_308 = None
        clone_78: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_619: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_78, [1024, 128, 32]);  clone_78 = None
        bmm_31: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_618, view_619);  view_618 = view_619 = None
        view_620: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_31, [256, 4, 128, 32]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_310: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_620, [0, 2, 1, 3]);  view_620 = None
        clone_79: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_621: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_79, [256, 128, -1]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_622: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_621, [32768, 128]);  view_621 = None
        permute_311: "f32[128, 128]" = torch.ops.aten.permute.default(arg713_1, [1, 0]);  arg713_1 = None
        addmm_231: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg714_1, view_622, permute_311);  arg714_1 = view_622 = permute_311 = None
        view_623: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_231, [256, 128, 128]);  addmm_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_602: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_229, [32768, 512])
        permute_301: "f32[512, 128]" = torch.ops.aten.permute.default(arg699_1, [1, 0]);  arg699_1 = None
        addmm_226: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg700_1, view_602, permute_301);  arg700_1 = view_602 = permute_301 = None
        view_603: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_226, [256, 128, 128]);  addmm_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_151: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_603, arg701_1);  view_603 = arg701_1 = None
        add_230: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_151, arg702_1);  mul_151 = arg702_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_233: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_623, add_230);  view_623 = add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_155: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_233, arg715_1);  add_233 = arg715_1 = None
        add_234: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_155, arg716_1);  mul_155 = arg716_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_624: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_234, [32768, 128])
        permute_312: "f32[128, 512]" = torch.ops.aten.permute.default(arg717_1, [1, 0]);  arg717_1 = None
        addmm_232: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg718_1, view_624, permute_312);  arg718_1 = view_624 = permute_312 = None
        view_625: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_232, [256, 128, 512]);  addmm_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_60: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_625);  view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_626: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_60, [32768, 512]);  relu_60 = None
        permute_313: "f32[512, 128]" = torch.ops.aten.permute.default(arg719_1, [1, 0]);  arg719_1 = None
        addmm_233: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg720_1, view_626, permute_313);  arg720_1 = view_626 = permute_313 = None
        view_627: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_233, [256, 128, 128]);  addmm_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_235: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_627, add_234);  view_627 = add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_156: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_235, arg721_1);  add_235 = arg721_1 = None
        add_236: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_156, arg722_1);  mul_156 = arg722_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_628: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_236, [32768, 128])
        permute_314: "f32[128, 512]" = torch.ops.aten.permute.default(arg723_1, [1, 0]);  arg723_1 = None
        addmm_234: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg724_1, view_628, permute_314);  arg724_1 = view_628 = permute_314 = None
        view_629: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_234, [256, 128, 512]);  addmm_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_61: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_629);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_630: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_61, [32768, 512]);  relu_61 = None
        permute_315: "f32[512, 128]" = torch.ops.aten.permute.default(arg725_1, [1, 0]);  arg725_1 = None
        addmm_235: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg726_1, view_630, permute_315);  arg726_1 = view_630 = permute_315 = None
        view_631: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_235, [256, 128, 128]);  addmm_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_237: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_631, add_236);  view_631 = add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_157: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_237, arg727_1);  add_237 = arg727_1 = None
        add_238: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_157, arg728_1);  mul_157 = arg728_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_632: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_238, [32768, 128])
        permute_316: "f32[128, 512]" = torch.ops.aten.permute.default(arg729_1, [1, 0]);  arg729_1 = None
        addmm_236: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg730_1, view_632, permute_316);  arg730_1 = view_632 = permute_316 = None
        view_633: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_236, [256, 128, 512]);  addmm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_62: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_633);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_634: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_62, [32768, 512]);  relu_62 = None
        permute_317: "f32[512, 128]" = torch.ops.aten.permute.default(arg731_1, [1, 0]);  arg731_1 = None
        addmm_237: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg732_1, view_634, permute_317);  arg732_1 = view_634 = permute_317 = None
        view_635: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_237, [256, 128, 128]);  addmm_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_239: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_635, add_238);  view_635 = add_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_158: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_239, arg733_1);  add_239 = arg733_1 = None
        add_240: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_158, arg734_1);  mul_158 = arg734_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_636: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_240, [32768, 128])
        permute_318: "f32[128, 512]" = torch.ops.aten.permute.default(arg735_1, [1, 0]);  arg735_1 = None
        addmm_238: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg736_1, view_636, permute_318);  arg736_1 = view_636 = permute_318 = None
        view_637: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_238, [256, 128, 512]);  addmm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_63: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_637);  view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_638: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_63, [32768, 512]);  relu_63 = None
        permute_319: "f32[512, 128]" = torch.ops.aten.permute.default(arg737_1, [1, 0]);  arg737_1 = None
        addmm_239: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg738_1, view_638, permute_319);  arg738_1 = view_638 = permute_319 = None
        view_639: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_239, [256, 128, 128]);  addmm_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_241: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_639, add_240);  view_639 = add_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_159: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_241, arg739_1);  add_241 = arg739_1 = None
        add_242: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_159, arg740_1);  mul_159 = arg740_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_640: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_242, [32768, 128]);  add_242 = None
        permute_320: "f32[128, 512]" = torch.ops.aten.permute.default(arg741_1, [1, 0]);  arg741_1 = None
        addmm_240: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg742_1, view_640, permute_320);  arg742_1 = view_640 = permute_320 = None
        view_641: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_240, [256, 128, 512]);  addmm_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_243: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_641, add_229);  view_641 = add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_160: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_243, arg743_1);  add_243 = arg743_1 = None
        add_244: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_160, arg744_1);  mul_160 = arg744_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_644: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_244, [32768, 512])
        permute_322: "f32[512, 128]" = torch.ops.aten.permute.default(arg749_1, [1, 0]);  arg749_1 = None
        addmm_242: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg750_1, view_644, permute_322);  arg750_1 = view_644 = permute_322 = None
        view_645: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_242, [256, 128, 128]);  addmm_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_162: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_645, arg751_1);  view_645 = arg751_1 = None
        add_246: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_162, arg752_1);  mul_162 = arg752_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_646: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_246, [32768, 128])
        permute_323: "f32[128, 128]" = torch.ops.aten.permute.default(arg753_1, [1, 0]);  arg753_1 = None
        addmm_243: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg754_1, view_646, permute_323);  arg754_1 = view_646 = permute_323 = None
        view_647: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_243, [256, 128, 128]);  addmm_243 = None
        view_648: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_647, [256, 128, -1, 32]);  view_647 = None
        permute_324: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_163: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_324, 0.4204482076268573);  permute_324 = None
        expand_65: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_163, [256, 4, 128, 32]);  mul_163 = None
        clone_81: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_655: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_81, [1024, 128, 32]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_649: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_246, [32768, 128]);  add_246 = None
        permute_325: "f32[128, 128]" = torch.ops.aten.permute.default(arg755_1, [1, 0]);  arg755_1 = None
        addmm_244: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg756_1, view_649, permute_325);  arg756_1 = view_649 = permute_325 = None
        view_650: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_244, [256, 128, 128]);  addmm_244 = None
        view_651: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_650, [256, 128, -1, 32]);  view_650 = None
        permute_326: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_651, [0, 2, 1, 3]);  view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_329: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_326, [0, 1, 3, 2]);  permute_326 = None
        mul_164: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_329, 0.4204482076268573);  permute_329 = None
        expand_66: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_164, [256, 4, 32, 128]);  mul_164 = None
        clone_82: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_656: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_82, [1024, 32, 128]);  clone_82 = None
        bmm_32: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_655, view_656);  view_655 = view_656 = None
        view_657: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [256, 4, 128, 128]);  bmm_32 = None
        full_default_50: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_50, full_default_49);  full_default_50 = full_default_49 = None
        add_247: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_657, where_32);  view_657 = where_32 = None
        eq_16: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_247, -inf)
        logical_not_32: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_16);  eq_16 = None
        any_17: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_32, -1, True);  logical_not_32 = None
        logical_not_33: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        full_default_51: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_16: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_247, [-1], True)
        sub_16: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_247, amax_16);  add_247 = amax_16 = None
        exp_16: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_17: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        where_33: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_33, full_default_51, div_16);  logical_not_33 = full_default_51 = div_16 = None
        expand_67: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_33, [256, 4, 128, 128]);  where_33 = None
        view_658: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_67, [1024, 128, 128]);  expand_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_652: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_244, [32768, 512])
        permute_327: "f32[512, 128]" = torch.ops.aten.permute.default(arg757_1, [1, 0]);  arg757_1 = None
        addmm_245: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg758_1, view_652, permute_327);  arg758_1 = view_652 = permute_327 = None
        view_653: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_245, [256, 128, 128]);  addmm_245 = None
        view_654: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_653, [256, 128, -1, 32]);  view_653 = None
        permute_328: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_68: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_328, [256, 4, 128, 32]);  permute_328 = None
        clone_83: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_659: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_83, [1024, 128, 32]);  clone_83 = None
        bmm_33: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_658, view_659);  view_658 = view_659 = None
        view_660: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_33, [256, 4, 128, 32]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_330: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_660, [0, 2, 1, 3]);  view_660 = None
        clone_84: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_661: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_84, [256, 128, -1]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_662: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_661, [32768, 128]);  view_661 = None
        permute_331: "f32[128, 128]" = torch.ops.aten.permute.default(arg759_1, [1, 0]);  arg759_1 = None
        addmm_246: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg760_1, view_662, permute_331);  arg760_1 = view_662 = permute_331 = None
        view_663: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_246, [256, 128, 128]);  addmm_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_642: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_244, [32768, 512])
        permute_321: "f32[512, 128]" = torch.ops.aten.permute.default(arg745_1, [1, 0]);  arg745_1 = None
        addmm_241: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg746_1, view_642, permute_321);  arg746_1 = view_642 = permute_321 = None
        view_643: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_241, [256, 128, 128]);  addmm_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_161: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_643, arg747_1);  view_643 = arg747_1 = None
        add_245: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_161, arg748_1);  mul_161 = arg748_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_248: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_663, add_245);  view_663 = add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_165: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_248, arg761_1);  add_248 = arg761_1 = None
        add_249: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_165, arg762_1);  mul_165 = arg762_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_664: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_249, [32768, 128])
        permute_332: "f32[128, 512]" = torch.ops.aten.permute.default(arg763_1, [1, 0]);  arg763_1 = None
        addmm_247: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg764_1, view_664, permute_332);  arg764_1 = view_664 = permute_332 = None
        view_665: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_247, [256, 128, 512]);  addmm_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_64: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_665);  view_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_666: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_64, [32768, 512]);  relu_64 = None
        permute_333: "f32[512, 128]" = torch.ops.aten.permute.default(arg765_1, [1, 0]);  arg765_1 = None
        addmm_248: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg766_1, view_666, permute_333);  arg766_1 = view_666 = permute_333 = None
        view_667: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_248, [256, 128, 128]);  addmm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_250: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_667, add_249);  view_667 = add_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_166: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_250, arg767_1);  add_250 = arg767_1 = None
        add_251: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_166, arg768_1);  mul_166 = arg768_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_668: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_251, [32768, 128])
        permute_334: "f32[128, 512]" = torch.ops.aten.permute.default(arg769_1, [1, 0]);  arg769_1 = None
        addmm_249: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg770_1, view_668, permute_334);  arg770_1 = view_668 = permute_334 = None
        view_669: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_249, [256, 128, 512]);  addmm_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_65: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_669);  view_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_670: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_65, [32768, 512]);  relu_65 = None
        permute_335: "f32[512, 128]" = torch.ops.aten.permute.default(arg771_1, [1, 0]);  arg771_1 = None
        addmm_250: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg772_1, view_670, permute_335);  arg772_1 = view_670 = permute_335 = None
        view_671: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_250, [256, 128, 128]);  addmm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_252: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_671, add_251);  view_671 = add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_167: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_252, arg773_1);  add_252 = arg773_1 = None
        add_253: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_167, arg774_1);  mul_167 = arg774_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_672: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_253, [32768, 128])
        permute_336: "f32[128, 512]" = torch.ops.aten.permute.default(arg775_1, [1, 0]);  arg775_1 = None
        addmm_251: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg776_1, view_672, permute_336);  arg776_1 = view_672 = permute_336 = None
        view_673: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_251, [256, 128, 512]);  addmm_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_66: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_673);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_674: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_66, [32768, 512]);  relu_66 = None
        permute_337: "f32[512, 128]" = torch.ops.aten.permute.default(arg777_1, [1, 0]);  arg777_1 = None
        addmm_252: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg778_1, view_674, permute_337);  arg778_1 = view_674 = permute_337 = None
        view_675: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_252, [256, 128, 128]);  addmm_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_254: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_675, add_253);  view_675 = add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_168: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_254, arg779_1);  add_254 = arg779_1 = None
        add_255: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_168, arg780_1);  mul_168 = arg780_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_676: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_255, [32768, 128])
        permute_338: "f32[128, 512]" = torch.ops.aten.permute.default(arg781_1, [1, 0]);  arg781_1 = None
        addmm_253: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg782_1, view_676, permute_338);  arg782_1 = view_676 = permute_338 = None
        view_677: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_253, [256, 128, 512]);  addmm_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_67: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_677);  view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_678: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_67, [32768, 512]);  relu_67 = None
        permute_339: "f32[512, 128]" = torch.ops.aten.permute.default(arg783_1, [1, 0]);  arg783_1 = None
        addmm_254: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg784_1, view_678, permute_339);  arg784_1 = view_678 = permute_339 = None
        view_679: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_254, [256, 128, 128]);  addmm_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_256: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_679, add_255);  view_679 = add_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_169: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_256, arg785_1);  add_256 = arg785_1 = None
        add_257: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_169, arg786_1);  mul_169 = arg786_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_680: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_257, [32768, 128]);  add_257 = None
        permute_340: "f32[128, 512]" = torch.ops.aten.permute.default(arg787_1, [1, 0]);  arg787_1 = None
        addmm_255: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg788_1, view_680, permute_340);  arg788_1 = view_680 = permute_340 = None
        view_681: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_255, [256, 128, 512]);  addmm_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_258: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_681, add_244);  view_681 = add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_170: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_258, arg789_1);  add_258 = arg789_1 = None
        add_259: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_170, arg790_1);  mul_170 = arg790_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_684: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_259, [32768, 512])
        permute_342: "f32[512, 128]" = torch.ops.aten.permute.default(arg795_1, [1, 0]);  arg795_1 = None
        addmm_257: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg796_1, view_684, permute_342);  arg796_1 = view_684 = permute_342 = None
        view_685: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_257, [256, 128, 128]);  addmm_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_172: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_685, arg797_1);  view_685 = arg797_1 = None
        add_261: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_172, arg798_1);  mul_172 = arg798_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_686: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_261, [32768, 128])
        permute_343: "f32[128, 128]" = torch.ops.aten.permute.default(arg799_1, [1, 0]);  arg799_1 = None
        addmm_258: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg800_1, view_686, permute_343);  arg800_1 = view_686 = permute_343 = None
        view_687: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_258, [256, 128, 128]);  addmm_258 = None
        view_688: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_687, [256, 128, -1, 32]);  view_687 = None
        permute_344: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_173: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_344, 0.4204482076268573);  permute_344 = None
        expand_69: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_173, [256, 4, 128, 32]);  mul_173 = None
        clone_86: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_695: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_86, [1024, 128, 32]);  clone_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_689: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_261, [32768, 128]);  add_261 = None
        permute_345: "f32[128, 128]" = torch.ops.aten.permute.default(arg801_1, [1, 0]);  arg801_1 = None
        addmm_259: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg802_1, view_689, permute_345);  arg802_1 = view_689 = permute_345 = None
        view_690: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_259, [256, 128, 128]);  addmm_259 = None
        view_691: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_690, [256, 128, -1, 32]);  view_690 = None
        permute_346: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_691, [0, 2, 1, 3]);  view_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_349: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_346, [0, 1, 3, 2]);  permute_346 = None
        mul_174: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_349, 0.4204482076268573);  permute_349 = None
        expand_70: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_174, [256, 4, 32, 128]);  mul_174 = None
        clone_87: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_696: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_87, [1024, 32, 128]);  clone_87 = None
        bmm_34: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_695, view_696);  view_695 = view_696 = None
        view_697: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [256, 4, 128, 128]);  bmm_34 = None
        full_default_53: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_53, full_default_52);  full_default_53 = full_default_52 = None
        add_262: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_697, where_34);  view_697 = where_34 = None
        eq_17: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_262, -inf)
        logical_not_34: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_17);  eq_17 = None
        any_18: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_34, -1, True);  logical_not_34 = None
        logical_not_35: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        full_default_54: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_17: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_262, [-1], True)
        sub_17: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_262, amax_17);  add_262 = amax_17 = None
        exp_17: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_18: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        where_35: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_35, full_default_54, div_17);  logical_not_35 = full_default_54 = div_17 = None
        expand_71: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_35, [256, 4, 128, 128]);  where_35 = None
        view_698: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_71, [1024, 128, 128]);  expand_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_692: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_259, [32768, 512])
        permute_347: "f32[512, 128]" = torch.ops.aten.permute.default(arg803_1, [1, 0]);  arg803_1 = None
        addmm_260: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg804_1, view_692, permute_347);  arg804_1 = view_692 = permute_347 = None
        view_693: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_260, [256, 128, 128]);  addmm_260 = None
        view_694: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_693, [256, 128, -1, 32]);  view_693 = None
        permute_348: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_72: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_348, [256, 4, 128, 32]);  permute_348 = None
        clone_88: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_699: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_88, [1024, 128, 32]);  clone_88 = None
        bmm_35: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_698, view_699);  view_698 = view_699 = None
        view_700: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_35, [256, 4, 128, 32]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_350: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
        clone_89: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_701: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_89, [256, 128, -1]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_702: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_701, [32768, 128]);  view_701 = None
        permute_351: "f32[128, 128]" = torch.ops.aten.permute.default(arg805_1, [1, 0]);  arg805_1 = None
        addmm_261: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg806_1, view_702, permute_351);  arg806_1 = view_702 = permute_351 = None
        view_703: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_261, [256, 128, 128]);  addmm_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_682: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_259, [32768, 512])
        permute_341: "f32[512, 128]" = torch.ops.aten.permute.default(arg791_1, [1, 0]);  arg791_1 = None
        addmm_256: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg792_1, view_682, permute_341);  arg792_1 = view_682 = permute_341 = None
        view_683: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_256, [256, 128, 128]);  addmm_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_171: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_683, arg793_1);  view_683 = arg793_1 = None
        add_260: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_171, arg794_1);  mul_171 = arg794_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_263: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_703, add_260);  view_703 = add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_175: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_263, arg807_1);  add_263 = arg807_1 = None
        add_264: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_175, arg808_1);  mul_175 = arg808_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_704: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_264, [32768, 128])
        permute_352: "f32[128, 512]" = torch.ops.aten.permute.default(arg809_1, [1, 0]);  arg809_1 = None
        addmm_262: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg810_1, view_704, permute_352);  arg810_1 = view_704 = permute_352 = None
        view_705: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_262, [256, 128, 512]);  addmm_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_68: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_705);  view_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_706: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_68, [32768, 512]);  relu_68 = None
        permute_353: "f32[512, 128]" = torch.ops.aten.permute.default(arg811_1, [1, 0]);  arg811_1 = None
        addmm_263: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg812_1, view_706, permute_353);  arg812_1 = view_706 = permute_353 = None
        view_707: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_263, [256, 128, 128]);  addmm_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_265: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_707, add_264);  view_707 = add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_176: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_265, arg813_1);  add_265 = arg813_1 = None
        add_266: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_176, arg814_1);  mul_176 = arg814_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_708: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_266, [32768, 128])
        permute_354: "f32[128, 512]" = torch.ops.aten.permute.default(arg815_1, [1, 0]);  arg815_1 = None
        addmm_264: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg816_1, view_708, permute_354);  arg816_1 = view_708 = permute_354 = None
        view_709: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_264, [256, 128, 512]);  addmm_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_69: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_709);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_710: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_69, [32768, 512]);  relu_69 = None
        permute_355: "f32[512, 128]" = torch.ops.aten.permute.default(arg817_1, [1, 0]);  arg817_1 = None
        addmm_265: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg818_1, view_710, permute_355);  arg818_1 = view_710 = permute_355 = None
        view_711: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_265, [256, 128, 128]);  addmm_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_267: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_711, add_266);  view_711 = add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_177: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_267, arg819_1);  add_267 = arg819_1 = None
        add_268: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_177, arg820_1);  mul_177 = arg820_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_712: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_268, [32768, 128])
        permute_356: "f32[128, 512]" = torch.ops.aten.permute.default(arg821_1, [1, 0]);  arg821_1 = None
        addmm_266: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg822_1, view_712, permute_356);  arg822_1 = view_712 = permute_356 = None
        view_713: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_266, [256, 128, 512]);  addmm_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_70: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_713);  view_713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_714: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_70, [32768, 512]);  relu_70 = None
        permute_357: "f32[512, 128]" = torch.ops.aten.permute.default(arg823_1, [1, 0]);  arg823_1 = None
        addmm_267: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg824_1, view_714, permute_357);  arg824_1 = view_714 = permute_357 = None
        view_715: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_267, [256, 128, 128]);  addmm_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_269: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_715, add_268);  view_715 = add_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_178: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_269, arg825_1);  add_269 = arg825_1 = None
        add_270: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_178, arg826_1);  mul_178 = arg826_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_716: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_270, [32768, 128])
        permute_358: "f32[128, 512]" = torch.ops.aten.permute.default(arg827_1, [1, 0]);  arg827_1 = None
        addmm_268: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg828_1, view_716, permute_358);  arg828_1 = view_716 = permute_358 = None
        view_717: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_268, [256, 128, 512]);  addmm_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_71: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_717);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_718: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_71, [32768, 512]);  relu_71 = None
        permute_359: "f32[512, 128]" = torch.ops.aten.permute.default(arg829_1, [1, 0]);  arg829_1 = None
        addmm_269: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg830_1, view_718, permute_359);  arg830_1 = view_718 = permute_359 = None
        view_719: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_269, [256, 128, 128]);  addmm_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_271: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_719, add_270);  view_719 = add_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_179: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_271, arg831_1);  add_271 = arg831_1 = None
        add_272: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_179, arg832_1);  mul_179 = arg832_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_720: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_272, [32768, 128]);  add_272 = None
        permute_360: "f32[128, 512]" = torch.ops.aten.permute.default(arg833_1, [1, 0]);  arg833_1 = None
        addmm_270: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg834_1, view_720, permute_360);  arg834_1 = view_720 = permute_360 = None
        view_721: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_270, [256, 128, 512]);  addmm_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_273: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_721, add_259);  view_721 = add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_180: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_273, arg835_1);  add_273 = arg835_1 = None
        add_274: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_180, arg836_1);  mul_180 = arg836_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_724: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_274, [32768, 512])
        permute_362: "f32[512, 128]" = torch.ops.aten.permute.default(arg841_1, [1, 0]);  arg841_1 = None
        addmm_272: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg842_1, view_724, permute_362);  arg842_1 = view_724 = permute_362 = None
        view_725: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_272, [256, 128, 128]);  addmm_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_182: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_725, arg843_1);  view_725 = arg843_1 = None
        add_276: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_182, arg844_1);  mul_182 = arg844_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_726: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_276, [32768, 128])
        permute_363: "f32[128, 128]" = torch.ops.aten.permute.default(arg845_1, [1, 0]);  arg845_1 = None
        addmm_273: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg846_1, view_726, permute_363);  arg846_1 = view_726 = permute_363 = None
        view_727: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_273, [256, 128, 128]);  addmm_273 = None
        view_728: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_727, [256, 128, -1, 32]);  view_727 = None
        permute_364: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_728, [0, 2, 1, 3]);  view_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_183: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_364, 0.4204482076268573);  permute_364 = None
        expand_73: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_183, [256, 4, 128, 32]);  mul_183 = None
        clone_91: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_735: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_91, [1024, 128, 32]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_729: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_276, [32768, 128]);  add_276 = None
        permute_365: "f32[128, 128]" = torch.ops.aten.permute.default(arg847_1, [1, 0]);  arg847_1 = None
        addmm_274: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg848_1, view_729, permute_365);  arg848_1 = view_729 = permute_365 = None
        view_730: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_274, [256, 128, 128]);  addmm_274 = None
        view_731: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_730, [256, 128, -1, 32]);  view_730 = None
        permute_366: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_731, [0, 2, 1, 3]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_369: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_366, [0, 1, 3, 2]);  permute_366 = None
        mul_184: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_369, 0.4204482076268573);  permute_369 = None
        expand_74: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_184, [256, 4, 32, 128]);  mul_184 = None
        clone_92: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_736: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_92, [1024, 32, 128]);  clone_92 = None
        bmm_36: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_735, view_736);  view_735 = view_736 = None
        view_737: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [256, 4, 128, 128]);  bmm_36 = None
        full_default_56: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_56, full_default_55);  full_default_56 = full_default_55 = None
        add_277: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_737, where_36);  view_737 = where_36 = None
        eq_18: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_277, -inf)
        logical_not_36: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_18);  eq_18 = None
        any_19: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_36, -1, True);  logical_not_36 = None
        logical_not_37: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        full_default_57: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_18: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_277, [-1], True)
        sub_18: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_277, amax_18);  add_277 = amax_18 = None
        exp_18: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_19: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        where_37: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_37, full_default_57, div_18);  logical_not_37 = full_default_57 = div_18 = None
        expand_75: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_37, [256, 4, 128, 128]);  where_37 = None
        view_738: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_75, [1024, 128, 128]);  expand_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_732: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_274, [32768, 512])
        permute_367: "f32[512, 128]" = torch.ops.aten.permute.default(arg849_1, [1, 0]);  arg849_1 = None
        addmm_275: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg850_1, view_732, permute_367);  arg850_1 = view_732 = permute_367 = None
        view_733: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_275, [256, 128, 128]);  addmm_275 = None
        view_734: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_733, [256, 128, -1, 32]);  view_733 = None
        permute_368: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_76: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_368, [256, 4, 128, 32]);  permute_368 = None
        clone_93: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_739: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_93, [1024, 128, 32]);  clone_93 = None
        bmm_37: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_738, view_739);  view_738 = view_739 = None
        view_740: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_37, [256, 4, 128, 32]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_370: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_740, [0, 2, 1, 3]);  view_740 = None
        clone_94: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_741: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_94, [256, 128, -1]);  clone_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_742: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_741, [32768, 128]);  view_741 = None
        permute_371: "f32[128, 128]" = torch.ops.aten.permute.default(arg851_1, [1, 0]);  arg851_1 = None
        addmm_276: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg852_1, view_742, permute_371);  arg852_1 = view_742 = permute_371 = None
        view_743: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_276, [256, 128, 128]);  addmm_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_722: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_274, [32768, 512])
        permute_361: "f32[512, 128]" = torch.ops.aten.permute.default(arg837_1, [1, 0]);  arg837_1 = None
        addmm_271: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg838_1, view_722, permute_361);  arg838_1 = view_722 = permute_361 = None
        view_723: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_271, [256, 128, 128]);  addmm_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_181: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_723, arg839_1);  view_723 = arg839_1 = None
        add_275: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_181, arg840_1);  mul_181 = arg840_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_278: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_743, add_275);  view_743 = add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_185: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_278, arg853_1);  add_278 = arg853_1 = None
        add_279: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_185, arg854_1);  mul_185 = arg854_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_744: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_279, [32768, 128])
        permute_372: "f32[128, 512]" = torch.ops.aten.permute.default(arg855_1, [1, 0]);  arg855_1 = None
        addmm_277: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg856_1, view_744, permute_372);  arg856_1 = view_744 = permute_372 = None
        view_745: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_277, [256, 128, 512]);  addmm_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_72: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_745);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_746: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_72, [32768, 512]);  relu_72 = None
        permute_373: "f32[512, 128]" = torch.ops.aten.permute.default(arg857_1, [1, 0]);  arg857_1 = None
        addmm_278: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg858_1, view_746, permute_373);  arg858_1 = view_746 = permute_373 = None
        view_747: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_278, [256, 128, 128]);  addmm_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_280: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_747, add_279);  view_747 = add_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_186: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_280, arg859_1);  add_280 = arg859_1 = None
        add_281: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_186, arg860_1);  mul_186 = arg860_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_748: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_281, [32768, 128])
        permute_374: "f32[128, 512]" = torch.ops.aten.permute.default(arg861_1, [1, 0]);  arg861_1 = None
        addmm_279: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg862_1, view_748, permute_374);  arg862_1 = view_748 = permute_374 = None
        view_749: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_279, [256, 128, 512]);  addmm_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_73: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_749);  view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_750: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_73, [32768, 512]);  relu_73 = None
        permute_375: "f32[512, 128]" = torch.ops.aten.permute.default(arg863_1, [1, 0]);  arg863_1 = None
        addmm_280: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg864_1, view_750, permute_375);  arg864_1 = view_750 = permute_375 = None
        view_751: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_280, [256, 128, 128]);  addmm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_282: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_751, add_281);  view_751 = add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_187: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_282, arg865_1);  add_282 = arg865_1 = None
        add_283: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_187, arg866_1);  mul_187 = arg866_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_752: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_283, [32768, 128])
        permute_376: "f32[128, 512]" = torch.ops.aten.permute.default(arg867_1, [1, 0]);  arg867_1 = None
        addmm_281: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg868_1, view_752, permute_376);  arg868_1 = view_752 = permute_376 = None
        view_753: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_281, [256, 128, 512]);  addmm_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_74: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_753);  view_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_754: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_74, [32768, 512]);  relu_74 = None
        permute_377: "f32[512, 128]" = torch.ops.aten.permute.default(arg869_1, [1, 0]);  arg869_1 = None
        addmm_282: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg870_1, view_754, permute_377);  arg870_1 = view_754 = permute_377 = None
        view_755: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_282, [256, 128, 128]);  addmm_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_284: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_755, add_283);  view_755 = add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_188: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_284, arg871_1);  add_284 = arg871_1 = None
        add_285: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_188, arg872_1);  mul_188 = arg872_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_756: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_285, [32768, 128])
        permute_378: "f32[128, 512]" = torch.ops.aten.permute.default(arg873_1, [1, 0]);  arg873_1 = None
        addmm_283: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg874_1, view_756, permute_378);  arg874_1 = view_756 = permute_378 = None
        view_757: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_283, [256, 128, 512]);  addmm_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_75: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_757);  view_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_758: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_75, [32768, 512]);  relu_75 = None
        permute_379: "f32[512, 128]" = torch.ops.aten.permute.default(arg875_1, [1, 0]);  arg875_1 = None
        addmm_284: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg876_1, view_758, permute_379);  arg876_1 = view_758 = permute_379 = None
        view_759: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_284, [256, 128, 128]);  addmm_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_286: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_759, add_285);  view_759 = add_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_189: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_286, arg877_1);  add_286 = arg877_1 = None
        add_287: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_189, arg878_1);  mul_189 = arg878_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_760: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_287, [32768, 128]);  add_287 = None
        permute_380: "f32[128, 512]" = torch.ops.aten.permute.default(arg879_1, [1, 0]);  arg879_1 = None
        addmm_285: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg880_1, view_760, permute_380);  arg880_1 = view_760 = permute_380 = None
        view_761: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_285, [256, 128, 512]);  addmm_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_288: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_761, add_274);  view_761 = add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_190: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_288, arg881_1);  add_288 = arg881_1 = None
        add_289: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_190, arg882_1);  mul_190 = arg882_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_764: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_289, [32768, 512])
        permute_382: "f32[512, 128]" = torch.ops.aten.permute.default(arg887_1, [1, 0]);  arg887_1 = None
        addmm_287: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg888_1, view_764, permute_382);  arg888_1 = view_764 = permute_382 = None
        view_765: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_287, [256, 128, 128]);  addmm_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_192: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_765, arg889_1);  view_765 = arg889_1 = None
        add_291: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_192, arg890_1);  mul_192 = arg890_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_766: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_291, [32768, 128])
        permute_383: "f32[128, 128]" = torch.ops.aten.permute.default(arg891_1, [1, 0]);  arg891_1 = None
        addmm_288: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg892_1, view_766, permute_383);  arg892_1 = view_766 = permute_383 = None
        view_767: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_288, [256, 128, 128]);  addmm_288 = None
        view_768: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_767, [256, 128, -1, 32]);  view_767 = None
        permute_384: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_768, [0, 2, 1, 3]);  view_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_193: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_384, 0.4204482076268573);  permute_384 = None
        expand_77: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_193, [256, 4, 128, 32]);  mul_193 = None
        clone_96: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_775: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_96, [1024, 128, 32]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_769: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_291, [32768, 128]);  add_291 = None
        permute_385: "f32[128, 128]" = torch.ops.aten.permute.default(arg893_1, [1, 0]);  arg893_1 = None
        addmm_289: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg894_1, view_769, permute_385);  arg894_1 = view_769 = permute_385 = None
        view_770: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_289, [256, 128, 128]);  addmm_289 = None
        view_771: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_770, [256, 128, -1, 32]);  view_770 = None
        permute_386: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_389: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_386, [0, 1, 3, 2]);  permute_386 = None
        mul_194: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_389, 0.4204482076268573);  permute_389 = None
        expand_78: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_194, [256, 4, 32, 128]);  mul_194 = None
        clone_97: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_776: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_97, [1024, 32, 128]);  clone_97 = None
        bmm_38: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_775, view_776);  view_775 = view_776 = None
        view_777: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [256, 4, 128, 128]);  bmm_38 = None
        full_default_59: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_38: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_59, full_default_58);  full_default_59 = full_default_58 = None
        add_292: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_777, where_38);  view_777 = where_38 = None
        eq_19: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_292, -inf)
        logical_not_38: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_19);  eq_19 = None
        any_20: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_38, -1, True);  logical_not_38 = None
        logical_not_39: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        full_default_60: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_19: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_292, [-1], True)
        sub_19: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_292, amax_19);  add_292 = amax_19 = None
        exp_19: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_20: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        where_39: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_39, full_default_60, div_19);  logical_not_39 = full_default_60 = div_19 = None
        expand_79: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_39, [256, 4, 128, 128]);  where_39 = None
        view_778: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_79, [1024, 128, 128]);  expand_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_772: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_289, [32768, 512])
        permute_387: "f32[512, 128]" = torch.ops.aten.permute.default(arg895_1, [1, 0]);  arg895_1 = None
        addmm_290: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg896_1, view_772, permute_387);  arg896_1 = view_772 = permute_387 = None
        view_773: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_290, [256, 128, 128]);  addmm_290 = None
        view_774: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_773, [256, 128, -1, 32]);  view_773 = None
        permute_388: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_80: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_388, [256, 4, 128, 32]);  permute_388 = None
        clone_98: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_779: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_98, [1024, 128, 32]);  clone_98 = None
        bmm_39: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_778, view_779);  view_778 = view_779 = None
        view_780: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_39, [256, 4, 128, 32]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_390: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
        clone_99: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_781: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_99, [256, 128, -1]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_782: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_781, [32768, 128]);  view_781 = None
        permute_391: "f32[128, 128]" = torch.ops.aten.permute.default(arg897_1, [1, 0]);  arg897_1 = None
        addmm_291: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg898_1, view_782, permute_391);  arg898_1 = view_782 = permute_391 = None
        view_783: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_291, [256, 128, 128]);  addmm_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_762: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_289, [32768, 512])
        permute_381: "f32[512, 128]" = torch.ops.aten.permute.default(arg883_1, [1, 0]);  arg883_1 = None
        addmm_286: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg884_1, view_762, permute_381);  arg884_1 = view_762 = permute_381 = None
        view_763: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_286, [256, 128, 128]);  addmm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_191: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_763, arg885_1);  view_763 = arg885_1 = None
        add_290: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_191, arg886_1);  mul_191 = arg886_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_293: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_783, add_290);  view_783 = add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_195: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_293, arg899_1);  add_293 = arg899_1 = None
        add_294: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_195, arg900_1);  mul_195 = arg900_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_784: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_294, [32768, 128])
        permute_392: "f32[128, 512]" = torch.ops.aten.permute.default(arg901_1, [1, 0]);  arg901_1 = None
        addmm_292: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg902_1, view_784, permute_392);  arg902_1 = view_784 = permute_392 = None
        view_785: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_292, [256, 128, 512]);  addmm_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_76: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_785);  view_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_786: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_76, [32768, 512]);  relu_76 = None
        permute_393: "f32[512, 128]" = torch.ops.aten.permute.default(arg903_1, [1, 0]);  arg903_1 = None
        addmm_293: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg904_1, view_786, permute_393);  arg904_1 = view_786 = permute_393 = None
        view_787: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_293, [256, 128, 128]);  addmm_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_295: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_787, add_294);  view_787 = add_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_196: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_295, arg905_1);  add_295 = arg905_1 = None
        add_296: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_196, arg906_1);  mul_196 = arg906_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_788: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_296, [32768, 128])
        permute_394: "f32[128, 512]" = torch.ops.aten.permute.default(arg907_1, [1, 0]);  arg907_1 = None
        addmm_294: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg908_1, view_788, permute_394);  arg908_1 = view_788 = permute_394 = None
        view_789: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_294, [256, 128, 512]);  addmm_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_77: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_789);  view_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_790: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_77, [32768, 512]);  relu_77 = None
        permute_395: "f32[512, 128]" = torch.ops.aten.permute.default(arg909_1, [1, 0]);  arg909_1 = None
        addmm_295: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg910_1, view_790, permute_395);  arg910_1 = view_790 = permute_395 = None
        view_791: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_295, [256, 128, 128]);  addmm_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_297: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_791, add_296);  view_791 = add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_197: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_297, arg911_1);  add_297 = arg911_1 = None
        add_298: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_197, arg912_1);  mul_197 = arg912_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_792: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_298, [32768, 128])
        permute_396: "f32[128, 512]" = torch.ops.aten.permute.default(arg913_1, [1, 0]);  arg913_1 = None
        addmm_296: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg914_1, view_792, permute_396);  arg914_1 = view_792 = permute_396 = None
        view_793: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_296, [256, 128, 512]);  addmm_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_78: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_793);  view_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_794: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_78, [32768, 512]);  relu_78 = None
        permute_397: "f32[512, 128]" = torch.ops.aten.permute.default(arg915_1, [1, 0]);  arg915_1 = None
        addmm_297: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg916_1, view_794, permute_397);  arg916_1 = view_794 = permute_397 = None
        view_795: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_297, [256, 128, 128]);  addmm_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_299: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_795, add_298);  view_795 = add_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_198: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_299, arg917_1);  add_299 = arg917_1 = None
        add_300: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_198, arg918_1);  mul_198 = arg918_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_796: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_300, [32768, 128])
        permute_398: "f32[128, 512]" = torch.ops.aten.permute.default(arg919_1, [1, 0]);  arg919_1 = None
        addmm_298: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg920_1, view_796, permute_398);  arg920_1 = view_796 = permute_398 = None
        view_797: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_298, [256, 128, 512]);  addmm_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_79: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_797);  view_797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_798: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_79, [32768, 512]);  relu_79 = None
        permute_399: "f32[512, 128]" = torch.ops.aten.permute.default(arg921_1, [1, 0]);  arg921_1 = None
        addmm_299: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg922_1, view_798, permute_399);  arg922_1 = view_798 = permute_399 = None
        view_799: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_299, [256, 128, 128]);  addmm_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_301: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_799, add_300);  view_799 = add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_199: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_301, arg923_1);  add_301 = arg923_1 = None
        add_302: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_199, arg924_1);  mul_199 = arg924_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_800: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_302, [32768, 128]);  add_302 = None
        permute_400: "f32[128, 512]" = torch.ops.aten.permute.default(arg925_1, [1, 0]);  arg925_1 = None
        addmm_300: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg926_1, view_800, permute_400);  arg926_1 = view_800 = permute_400 = None
        view_801: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_300, [256, 128, 512]);  addmm_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_303: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_801, add_289);  view_801 = add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_200: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_303, arg927_1);  add_303 = arg927_1 = None
        add_304: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_200, arg928_1);  mul_200 = arg928_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_804: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_304, [32768, 512])
        permute_402: "f32[512, 128]" = torch.ops.aten.permute.default(arg933_1, [1, 0]);  arg933_1 = None
        addmm_302: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg934_1, view_804, permute_402);  arg934_1 = view_804 = permute_402 = None
        view_805: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_302, [256, 128, 128]);  addmm_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_202: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_805, arg935_1);  view_805 = arg935_1 = None
        add_306: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_202, arg936_1);  mul_202 = arg936_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_806: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_306, [32768, 128])
        permute_403: "f32[128, 128]" = torch.ops.aten.permute.default(arg937_1, [1, 0]);  arg937_1 = None
        addmm_303: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg938_1, view_806, permute_403);  arg938_1 = view_806 = permute_403 = None
        view_807: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_303, [256, 128, 128]);  addmm_303 = None
        view_808: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_807, [256, 128, -1, 32]);  view_807 = None
        permute_404: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_808, [0, 2, 1, 3]);  view_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_203: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_404, 0.4204482076268573);  permute_404 = None
        expand_81: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_203, [256, 4, 128, 32]);  mul_203 = None
        clone_101: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_815: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_101, [1024, 128, 32]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_809: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_306, [32768, 128]);  add_306 = None
        permute_405: "f32[128, 128]" = torch.ops.aten.permute.default(arg939_1, [1, 0]);  arg939_1 = None
        addmm_304: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg940_1, view_809, permute_405);  arg940_1 = view_809 = permute_405 = None
        view_810: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_304, [256, 128, 128]);  addmm_304 = None
        view_811: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_810, [256, 128, -1, 32]);  view_810 = None
        permute_406: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_811, [0, 2, 1, 3]);  view_811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_409: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_406, [0, 1, 3, 2]);  permute_406 = None
        mul_204: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_409, 0.4204482076268573);  permute_409 = None
        expand_82: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_204, [256, 4, 32, 128]);  mul_204 = None
        clone_102: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_816: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_102, [1024, 32, 128]);  clone_102 = None
        bmm_40: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_815, view_816);  view_815 = view_816 = None
        view_817: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [256, 4, 128, 128]);  bmm_40 = None
        full_default_62: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_62, full_default_61);  full_default_62 = full_default_61 = None
        add_307: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_817, where_40);  view_817 = where_40 = None
        eq_20: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_307, -inf)
        logical_not_40: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_20);  eq_20 = None
        any_21: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_40, -1, True);  logical_not_40 = None
        logical_not_41: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        full_default_63: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_20: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_307, [-1], True)
        sub_20: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_307, amax_20);  add_307 = amax_20 = None
        exp_20: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_21: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        where_41: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_41, full_default_63, div_20);  logical_not_41 = full_default_63 = div_20 = None
        expand_83: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_41, [256, 4, 128, 128]);  where_41 = None
        view_818: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_83, [1024, 128, 128]);  expand_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_812: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_304, [32768, 512])
        permute_407: "f32[512, 128]" = torch.ops.aten.permute.default(arg941_1, [1, 0]);  arg941_1 = None
        addmm_305: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg942_1, view_812, permute_407);  arg942_1 = view_812 = permute_407 = None
        view_813: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_305, [256, 128, 128]);  addmm_305 = None
        view_814: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_813, [256, 128, -1, 32]);  view_813 = None
        permute_408: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_814, [0, 2, 1, 3]);  view_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_84: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_408, [256, 4, 128, 32]);  permute_408 = None
        clone_103: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_819: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_103, [1024, 128, 32]);  clone_103 = None
        bmm_41: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_818, view_819);  view_818 = view_819 = None
        view_820: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_41, [256, 4, 128, 32]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_410: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_820, [0, 2, 1, 3]);  view_820 = None
        clone_104: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_821: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_104, [256, 128, -1]);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_822: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_821, [32768, 128]);  view_821 = None
        permute_411: "f32[128, 128]" = torch.ops.aten.permute.default(arg943_1, [1, 0]);  arg943_1 = None
        addmm_306: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg944_1, view_822, permute_411);  arg944_1 = view_822 = permute_411 = None
        view_823: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_306, [256, 128, 128]);  addmm_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_802: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_304, [32768, 512])
        permute_401: "f32[512, 128]" = torch.ops.aten.permute.default(arg929_1, [1, 0]);  arg929_1 = None
        addmm_301: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg930_1, view_802, permute_401);  arg930_1 = view_802 = permute_401 = None
        view_803: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_301, [256, 128, 128]);  addmm_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_201: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_803, arg931_1);  view_803 = arg931_1 = None
        add_305: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_201, arg932_1);  mul_201 = arg932_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_308: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_823, add_305);  view_823 = add_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_205: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_308, arg945_1);  add_308 = arg945_1 = None
        add_309: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_205, arg946_1);  mul_205 = arg946_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_824: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_309, [32768, 128])
        permute_412: "f32[128, 512]" = torch.ops.aten.permute.default(arg947_1, [1, 0]);  arg947_1 = None
        addmm_307: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg948_1, view_824, permute_412);  arg948_1 = view_824 = permute_412 = None
        view_825: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_307, [256, 128, 512]);  addmm_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_80: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_825);  view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_826: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_80, [32768, 512]);  relu_80 = None
        permute_413: "f32[512, 128]" = torch.ops.aten.permute.default(arg949_1, [1, 0]);  arg949_1 = None
        addmm_308: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg950_1, view_826, permute_413);  arg950_1 = view_826 = permute_413 = None
        view_827: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_308, [256, 128, 128]);  addmm_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_310: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_827, add_309);  view_827 = add_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_206: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_310, arg951_1);  add_310 = arg951_1 = None
        add_311: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_206, arg952_1);  mul_206 = arg952_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_828: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_311, [32768, 128])
        permute_414: "f32[128, 512]" = torch.ops.aten.permute.default(arg953_1, [1, 0]);  arg953_1 = None
        addmm_309: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg954_1, view_828, permute_414);  arg954_1 = view_828 = permute_414 = None
        view_829: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_309, [256, 128, 512]);  addmm_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_81: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_829);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_830: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_81, [32768, 512]);  relu_81 = None
        permute_415: "f32[512, 128]" = torch.ops.aten.permute.default(arg955_1, [1, 0]);  arg955_1 = None
        addmm_310: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg956_1, view_830, permute_415);  arg956_1 = view_830 = permute_415 = None
        view_831: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_310, [256, 128, 128]);  addmm_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_312: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_831, add_311);  view_831 = add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_207: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_312, arg957_1);  add_312 = arg957_1 = None
        add_313: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_207, arg958_1);  mul_207 = arg958_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_832: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_313, [32768, 128])
        permute_416: "f32[128, 512]" = torch.ops.aten.permute.default(arg959_1, [1, 0]);  arg959_1 = None
        addmm_311: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg960_1, view_832, permute_416);  arg960_1 = view_832 = permute_416 = None
        view_833: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_311, [256, 128, 512]);  addmm_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_82: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_833);  view_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_834: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_82, [32768, 512]);  relu_82 = None
        permute_417: "f32[512, 128]" = torch.ops.aten.permute.default(arg961_1, [1, 0]);  arg961_1 = None
        addmm_312: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg962_1, view_834, permute_417);  arg962_1 = view_834 = permute_417 = None
        view_835: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_312, [256, 128, 128]);  addmm_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_314: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_835, add_313);  view_835 = add_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_208: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_314, arg963_1);  add_314 = arg963_1 = None
        add_315: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_208, arg964_1);  mul_208 = arg964_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_836: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_315, [32768, 128])
        permute_418: "f32[128, 512]" = torch.ops.aten.permute.default(arg965_1, [1, 0]);  arg965_1 = None
        addmm_313: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg966_1, view_836, permute_418);  arg966_1 = view_836 = permute_418 = None
        view_837: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_313, [256, 128, 512]);  addmm_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_83: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_837);  view_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_838: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_83, [32768, 512]);  relu_83 = None
        permute_419: "f32[512, 128]" = torch.ops.aten.permute.default(arg967_1, [1, 0]);  arg967_1 = None
        addmm_314: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg968_1, view_838, permute_419);  arg968_1 = view_838 = permute_419 = None
        view_839: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_314, [256, 128, 128]);  addmm_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_316: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_839, add_315);  view_839 = add_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_209: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_316, arg969_1);  add_316 = arg969_1 = None
        add_317: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_209, arg970_1);  mul_209 = arg970_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_840: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_317, [32768, 128]);  add_317 = None
        permute_420: "f32[128, 512]" = torch.ops.aten.permute.default(arg971_1, [1, 0]);  arg971_1 = None
        addmm_315: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg972_1, view_840, permute_420);  arg972_1 = view_840 = permute_420 = None
        view_841: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_315, [256, 128, 512]);  addmm_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_318: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_841, add_304);  view_841 = add_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_210: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_318, arg973_1);  add_318 = arg973_1 = None
        add_319: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_210, arg974_1);  mul_210 = arg974_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_844: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_319, [32768, 512])
        permute_422: "f32[512, 128]" = torch.ops.aten.permute.default(arg979_1, [1, 0]);  arg979_1 = None
        addmm_317: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg980_1, view_844, permute_422);  arg980_1 = view_844 = permute_422 = None
        view_845: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_317, [256, 128, 128]);  addmm_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_212: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_845, arg981_1);  view_845 = arg981_1 = None
        add_321: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_212, arg982_1);  mul_212 = arg982_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_846: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_321, [32768, 128])
        permute_423: "f32[128, 128]" = torch.ops.aten.permute.default(arg983_1, [1, 0]);  arg983_1 = None
        addmm_318: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg984_1, view_846, permute_423);  arg984_1 = view_846 = permute_423 = None
        view_847: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_318, [256, 128, 128]);  addmm_318 = None
        view_848: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_847, [256, 128, -1, 32]);  view_847 = None
        permute_424: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_848, [0, 2, 1, 3]);  view_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_213: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_424, 0.4204482076268573);  permute_424 = None
        expand_85: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_213, [256, 4, 128, 32]);  mul_213 = None
        clone_106: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_855: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_106, [1024, 128, 32]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_849: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_321, [32768, 128]);  add_321 = None
        permute_425: "f32[128, 128]" = torch.ops.aten.permute.default(arg985_1, [1, 0]);  arg985_1 = None
        addmm_319: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg986_1, view_849, permute_425);  arg986_1 = view_849 = permute_425 = None
        view_850: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_319, [256, 128, 128]);  addmm_319 = None
        view_851: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_850, [256, 128, -1, 32]);  view_850 = None
        permute_426: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_851, [0, 2, 1, 3]);  view_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_429: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_426, [0, 1, 3, 2]);  permute_426 = None
        mul_214: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_429, 0.4204482076268573);  permute_429 = None
        expand_86: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_214, [256, 4, 32, 128]);  mul_214 = None
        clone_107: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_856: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_107, [1024, 32, 128]);  clone_107 = None
        bmm_42: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_855, view_856);  view_855 = view_856 = None
        view_857: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [256, 4, 128, 128]);  bmm_42 = None
        full_default_65: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_42: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_65, full_default_64);  full_default_65 = full_default_64 = None
        add_322: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_857, where_42);  view_857 = where_42 = None
        eq_21: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_322, -inf)
        logical_not_42: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_21);  eq_21 = None
        any_22: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_42, -1, True);  logical_not_42 = None
        logical_not_43: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        full_default_66: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_21: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_322, [-1], True)
        sub_21: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_322, amax_21);  add_322 = amax_21 = None
        exp_21: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_22: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        where_43: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_43, full_default_66, div_21);  logical_not_43 = full_default_66 = div_21 = None
        expand_87: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_43, [256, 4, 128, 128]);  where_43 = None
        view_858: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_87, [1024, 128, 128]);  expand_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_852: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_319, [32768, 512])
        permute_427: "f32[512, 128]" = torch.ops.aten.permute.default(arg987_1, [1, 0]);  arg987_1 = None
        addmm_320: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg988_1, view_852, permute_427);  arg988_1 = view_852 = permute_427 = None
        view_853: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_320, [256, 128, 128]);  addmm_320 = None
        view_854: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_853, [256, 128, -1, 32]);  view_853 = None
        permute_428: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_88: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_428, [256, 4, 128, 32]);  permute_428 = None
        clone_108: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_859: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_108, [1024, 128, 32]);  clone_108 = None
        bmm_43: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_858, view_859);  view_858 = view_859 = None
        view_860: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_43, [256, 4, 128, 32]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_430: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_860, [0, 2, 1, 3]);  view_860 = None
        clone_109: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_430, memory_format = torch.contiguous_format);  permute_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_861: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_109, [256, 128, -1]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_862: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_861, [32768, 128]);  view_861 = None
        permute_431: "f32[128, 128]" = torch.ops.aten.permute.default(arg989_1, [1, 0]);  arg989_1 = None
        addmm_321: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg990_1, view_862, permute_431);  arg990_1 = view_862 = permute_431 = None
        view_863: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_321, [256, 128, 128]);  addmm_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_842: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_319, [32768, 512])
        permute_421: "f32[512, 128]" = torch.ops.aten.permute.default(arg975_1, [1, 0]);  arg975_1 = None
        addmm_316: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg976_1, view_842, permute_421);  arg976_1 = view_842 = permute_421 = None
        view_843: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_316, [256, 128, 128]);  addmm_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_211: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_843, arg977_1);  view_843 = arg977_1 = None
        add_320: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_211, arg978_1);  mul_211 = arg978_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_323: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_863, add_320);  view_863 = add_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_215: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_323, arg991_1);  add_323 = arg991_1 = None
        add_324: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_215, arg992_1);  mul_215 = arg992_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_864: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_324, [32768, 128])
        permute_432: "f32[128, 512]" = torch.ops.aten.permute.default(arg993_1, [1, 0]);  arg993_1 = None
        addmm_322: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg994_1, view_864, permute_432);  arg994_1 = view_864 = permute_432 = None
        view_865: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_322, [256, 128, 512]);  addmm_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_84: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_865);  view_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_866: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_84, [32768, 512]);  relu_84 = None
        permute_433: "f32[512, 128]" = torch.ops.aten.permute.default(arg995_1, [1, 0]);  arg995_1 = None
        addmm_323: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg996_1, view_866, permute_433);  arg996_1 = view_866 = permute_433 = None
        view_867: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_323, [256, 128, 128]);  addmm_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_325: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_867, add_324);  view_867 = add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_216: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_325, arg997_1);  add_325 = arg997_1 = None
        add_326: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_216, arg998_1);  mul_216 = arg998_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_868: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_326, [32768, 128])
        permute_434: "f32[128, 512]" = torch.ops.aten.permute.default(arg999_1, [1, 0]);  arg999_1 = None
        addmm_324: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1000_1, view_868, permute_434);  arg1000_1 = view_868 = permute_434 = None
        view_869: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_324, [256, 128, 512]);  addmm_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_85: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_869);  view_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_870: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_85, [32768, 512]);  relu_85 = None
        permute_435: "f32[512, 128]" = torch.ops.aten.permute.default(arg1001_1, [1, 0]);  arg1001_1 = None
        addmm_325: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1002_1, view_870, permute_435);  arg1002_1 = view_870 = permute_435 = None
        view_871: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_325, [256, 128, 128]);  addmm_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_327: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_871, add_326);  view_871 = add_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_217: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_327, arg1003_1);  add_327 = arg1003_1 = None
        add_328: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_217, arg1004_1);  mul_217 = arg1004_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_872: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_328, [32768, 128])
        permute_436: "f32[128, 512]" = torch.ops.aten.permute.default(arg1005_1, [1, 0]);  arg1005_1 = None
        addmm_326: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1006_1, view_872, permute_436);  arg1006_1 = view_872 = permute_436 = None
        view_873: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_326, [256, 128, 512]);  addmm_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_86: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_873);  view_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_874: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_86, [32768, 512]);  relu_86 = None
        permute_437: "f32[512, 128]" = torch.ops.aten.permute.default(arg1007_1, [1, 0]);  arg1007_1 = None
        addmm_327: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1008_1, view_874, permute_437);  arg1008_1 = view_874 = permute_437 = None
        view_875: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_327, [256, 128, 128]);  addmm_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_329: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_875, add_328);  view_875 = add_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_218: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_329, arg1009_1);  add_329 = arg1009_1 = None
        add_330: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_218, arg1010_1);  mul_218 = arg1010_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_876: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_330, [32768, 128])
        permute_438: "f32[128, 512]" = torch.ops.aten.permute.default(arg1011_1, [1, 0]);  arg1011_1 = None
        addmm_328: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1012_1, view_876, permute_438);  arg1012_1 = view_876 = permute_438 = None
        view_877: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_328, [256, 128, 512]);  addmm_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_87: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_877);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_878: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_87, [32768, 512]);  relu_87 = None
        permute_439: "f32[512, 128]" = torch.ops.aten.permute.default(arg1013_1, [1, 0]);  arg1013_1 = None
        addmm_329: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1014_1, view_878, permute_439);  arg1014_1 = view_878 = permute_439 = None
        view_879: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_329, [256, 128, 128]);  addmm_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_331: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_879, add_330);  view_879 = add_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_219: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_331, arg1015_1);  add_331 = arg1015_1 = None
        add_332: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_219, arg1016_1);  mul_219 = arg1016_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_880: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_332, [32768, 128]);  add_332 = None
        permute_440: "f32[128, 512]" = torch.ops.aten.permute.default(arg1017_1, [1, 0]);  arg1017_1 = None
        addmm_330: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1018_1, view_880, permute_440);  arg1018_1 = view_880 = permute_440 = None
        view_881: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_330, [256, 128, 512]);  addmm_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_333: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_881, add_319);  view_881 = add_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_220: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_333, arg1019_1);  add_333 = arg1019_1 = None
        add_334: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_220, arg1020_1);  mul_220 = arg1020_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_884: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_334, [32768, 512])
        permute_442: "f32[512, 128]" = torch.ops.aten.permute.default(arg1025_1, [1, 0]);  arg1025_1 = None
        addmm_332: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1026_1, view_884, permute_442);  arg1026_1 = view_884 = permute_442 = None
        view_885: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_332, [256, 128, 128]);  addmm_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_222: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_885, arg1027_1);  view_885 = arg1027_1 = None
        add_336: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_222, arg1028_1);  mul_222 = arg1028_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_886: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_336, [32768, 128])
        permute_443: "f32[128, 128]" = torch.ops.aten.permute.default(arg1029_1, [1, 0]);  arg1029_1 = None
        addmm_333: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1030_1, view_886, permute_443);  arg1030_1 = view_886 = permute_443 = None
        view_887: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_333, [256, 128, 128]);  addmm_333 = None
        view_888: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_887, [256, 128, -1, 32]);  view_887 = None
        permute_444: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_888, [0, 2, 1, 3]);  view_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_223: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_444, 0.4204482076268573);  permute_444 = None
        expand_89: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_223, [256, 4, 128, 32]);  mul_223 = None
        clone_111: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_895: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_111, [1024, 128, 32]);  clone_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_889: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_336, [32768, 128]);  add_336 = None
        permute_445: "f32[128, 128]" = torch.ops.aten.permute.default(arg1031_1, [1, 0]);  arg1031_1 = None
        addmm_334: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1032_1, view_889, permute_445);  arg1032_1 = view_889 = permute_445 = None
        view_890: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_334, [256, 128, 128]);  addmm_334 = None
        view_891: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_890, [256, 128, -1, 32]);  view_890 = None
        permute_446: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_891, [0, 2, 1, 3]);  view_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_449: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_446, [0, 1, 3, 2]);  permute_446 = None
        mul_224: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_449, 0.4204482076268573);  permute_449 = None
        expand_90: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_224, [256, 4, 32, 128]);  mul_224 = None
        clone_112: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_896: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_112, [1024, 32, 128]);  clone_112 = None
        bmm_44: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_895, view_896);  view_895 = view_896 = None
        view_897: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [256, 4, 128, 128]);  bmm_44 = None
        full_default_68: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_67: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_68, full_default_67);  full_default_68 = full_default_67 = None
        add_337: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_897, where_44);  view_897 = where_44 = None
        eq_22: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_337, -inf)
        logical_not_44: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_22);  eq_22 = None
        any_23: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_44, -1, True);  logical_not_44 = None
        logical_not_45: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        full_default_69: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_22: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_337, [-1], True)
        sub_22: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_337, amax_22);  add_337 = amax_22 = None
        exp_22: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_23: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        where_45: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_45, full_default_69, div_22);  logical_not_45 = full_default_69 = div_22 = None
        expand_91: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_45, [256, 4, 128, 128]);  where_45 = None
        view_898: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_91, [1024, 128, 128]);  expand_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_892: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_334, [32768, 512])
        permute_447: "f32[512, 128]" = torch.ops.aten.permute.default(arg1033_1, [1, 0]);  arg1033_1 = None
        addmm_335: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1034_1, view_892, permute_447);  arg1034_1 = view_892 = permute_447 = None
        view_893: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_335, [256, 128, 128]);  addmm_335 = None
        view_894: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_893, [256, 128, -1, 32]);  view_893 = None
        permute_448: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_894, [0, 2, 1, 3]);  view_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_92: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_448, [256, 4, 128, 32]);  permute_448 = None
        clone_113: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_899: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_113, [1024, 128, 32]);  clone_113 = None
        bmm_45: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_898, view_899);  view_898 = view_899 = None
        view_900: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_45, [256, 4, 128, 32]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_450: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_900, [0, 2, 1, 3]);  view_900 = None
        clone_114: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_901: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_114, [256, 128, -1]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_902: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_901, [32768, 128]);  view_901 = None
        permute_451: "f32[128, 128]" = torch.ops.aten.permute.default(arg1035_1, [1, 0]);  arg1035_1 = None
        addmm_336: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1036_1, view_902, permute_451);  arg1036_1 = view_902 = permute_451 = None
        view_903: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_336, [256, 128, 128]);  addmm_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_882: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_334, [32768, 512])
        permute_441: "f32[512, 128]" = torch.ops.aten.permute.default(arg1021_1, [1, 0]);  arg1021_1 = None
        addmm_331: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1022_1, view_882, permute_441);  arg1022_1 = view_882 = permute_441 = None
        view_883: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_331, [256, 128, 128]);  addmm_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_221: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_883, arg1023_1);  view_883 = arg1023_1 = None
        add_335: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_221, arg1024_1);  mul_221 = arg1024_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_338: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_903, add_335);  view_903 = add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_225: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_338, arg1037_1);  add_338 = arg1037_1 = None
        add_339: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_225, arg1038_1);  mul_225 = arg1038_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_904: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_339, [32768, 128])
        permute_452: "f32[128, 512]" = torch.ops.aten.permute.default(arg1039_1, [1, 0]);  arg1039_1 = None
        addmm_337: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1040_1, view_904, permute_452);  arg1040_1 = view_904 = permute_452 = None
        view_905: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_337, [256, 128, 512]);  addmm_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_88: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_905);  view_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_906: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_88, [32768, 512]);  relu_88 = None
        permute_453: "f32[512, 128]" = torch.ops.aten.permute.default(arg1041_1, [1, 0]);  arg1041_1 = None
        addmm_338: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1042_1, view_906, permute_453);  arg1042_1 = view_906 = permute_453 = None
        view_907: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_338, [256, 128, 128]);  addmm_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_340: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_907, add_339);  view_907 = add_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_226: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_340, arg1043_1);  add_340 = arg1043_1 = None
        add_341: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_226, arg1044_1);  mul_226 = arg1044_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_908: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_341, [32768, 128])
        permute_454: "f32[128, 512]" = torch.ops.aten.permute.default(arg1045_1, [1, 0]);  arg1045_1 = None
        addmm_339: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1046_1, view_908, permute_454);  arg1046_1 = view_908 = permute_454 = None
        view_909: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_339, [256, 128, 512]);  addmm_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_89: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_909);  view_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_910: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_89, [32768, 512]);  relu_89 = None
        permute_455: "f32[512, 128]" = torch.ops.aten.permute.default(arg1047_1, [1, 0]);  arg1047_1 = None
        addmm_340: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1048_1, view_910, permute_455);  arg1048_1 = view_910 = permute_455 = None
        view_911: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_340, [256, 128, 128]);  addmm_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_342: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_911, add_341);  view_911 = add_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_227: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_342, arg1049_1);  add_342 = arg1049_1 = None
        add_343: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_227, arg1050_1);  mul_227 = arg1050_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_912: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_343, [32768, 128])
        permute_456: "f32[128, 512]" = torch.ops.aten.permute.default(arg1051_1, [1, 0]);  arg1051_1 = None
        addmm_341: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1052_1, view_912, permute_456);  arg1052_1 = view_912 = permute_456 = None
        view_913: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_341, [256, 128, 512]);  addmm_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_90: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_913);  view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_914: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_90, [32768, 512]);  relu_90 = None
        permute_457: "f32[512, 128]" = torch.ops.aten.permute.default(arg1053_1, [1, 0]);  arg1053_1 = None
        addmm_342: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1054_1, view_914, permute_457);  arg1054_1 = view_914 = permute_457 = None
        view_915: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_342, [256, 128, 128]);  addmm_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_344: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_915, add_343);  view_915 = add_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_228: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_344, arg1055_1);  add_344 = arg1055_1 = None
        add_345: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_228, arg1056_1);  mul_228 = arg1056_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_916: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_345, [32768, 128])
        permute_458: "f32[128, 512]" = torch.ops.aten.permute.default(arg1057_1, [1, 0]);  arg1057_1 = None
        addmm_343: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1058_1, view_916, permute_458);  arg1058_1 = view_916 = permute_458 = None
        view_917: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_343, [256, 128, 512]);  addmm_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_91: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_917);  view_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_918: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_91, [32768, 512]);  relu_91 = None
        permute_459: "f32[512, 128]" = torch.ops.aten.permute.default(arg1059_1, [1, 0]);  arg1059_1 = None
        addmm_344: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1060_1, view_918, permute_459);  arg1060_1 = view_918 = permute_459 = None
        view_919: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_344, [256, 128, 128]);  addmm_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_346: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_919, add_345);  view_919 = add_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_229: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_346, arg1061_1);  add_346 = arg1061_1 = None
        add_347: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_229, arg1062_1);  mul_229 = arg1062_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_920: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_347, [32768, 128]);  add_347 = None
        permute_460: "f32[128, 512]" = torch.ops.aten.permute.default(arg1063_1, [1, 0]);  arg1063_1 = None
        addmm_345: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1064_1, view_920, permute_460);  arg1064_1 = view_920 = permute_460 = None
        view_921: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_345, [256, 128, 512]);  addmm_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_348: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_921, add_334);  view_921 = add_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_230: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_348, arg1065_1);  add_348 = arg1065_1 = None
        add_349: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_230, arg1066_1);  mul_230 = arg1066_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_924: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_349, [32768, 512])
        permute_462: "f32[512, 128]" = torch.ops.aten.permute.default(arg1071_1, [1, 0]);  arg1071_1 = None
        addmm_347: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1072_1, view_924, permute_462);  arg1072_1 = view_924 = permute_462 = None
        view_925: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_347, [256, 128, 128]);  addmm_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_232: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_925, arg1073_1);  view_925 = arg1073_1 = None
        add_351: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_232, arg1074_1);  mul_232 = arg1074_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_926: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_351, [32768, 128])
        permute_463: "f32[128, 128]" = torch.ops.aten.permute.default(arg1075_1, [1, 0]);  arg1075_1 = None
        addmm_348: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1076_1, view_926, permute_463);  arg1076_1 = view_926 = permute_463 = None
        view_927: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_348, [256, 128, 128]);  addmm_348 = None
        view_928: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_927, [256, 128, -1, 32]);  view_927 = None
        permute_464: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_928, [0, 2, 1, 3]);  view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_233: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_464, 0.4204482076268573);  permute_464 = None
        expand_93: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_233, [256, 4, 128, 32]);  mul_233 = None
        clone_116: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_935: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_116, [1024, 128, 32]);  clone_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        view_929: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_351, [32768, 128]);  add_351 = None
        permute_465: "f32[128, 128]" = torch.ops.aten.permute.default(arg1077_1, [1, 0]);  arg1077_1 = None
        addmm_349: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1078_1, view_929, permute_465);  arg1078_1 = view_929 = permute_465 = None
        view_930: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_349, [256, 128, 128]);  addmm_349 = None
        view_931: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_930, [256, 128, -1, 32]);  view_930 = None
        permute_466: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_931, [0, 2, 1, 3]);  view_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_469: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_466, [0, 1, 3, 2]);  permute_466 = None
        mul_234: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_469, 0.4204482076268573);  permute_469 = None
        expand_94: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_234, [256, 4, 32, 128]);  mul_234 = None
        clone_117: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_936: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_117, [1024, 32, 128]);  clone_117 = None
        bmm_46: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_935, view_936);  view_935 = view_936 = None
        view_937: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [256, 4, 128, 128]);  bmm_46 = None
        full_default_71: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_70: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_46: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_71, full_default_70);  expand = full_default_71 = full_default_70 = None
        add_352: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_937, where_46);  view_937 = where_46 = None
        eq_23: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_352, -inf)
        logical_not_46: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_23);  eq_23 = None
        any_24: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_46, -1, True);  logical_not_46 = None
        logical_not_47: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        full_default_72: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_23: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_352, [-1], True)
        sub_23: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_352, amax_23);  add_352 = amax_23 = None
        exp_23: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_24: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        where_47: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_47, full_default_72, div_23);  logical_not_47 = full_default_72 = div_23 = None
        expand_95: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_47, [256, 4, 128, 128]);  where_47 = None
        view_938: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_95, [1024, 128, 128]);  expand_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        view_932: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_349, [32768, 512])
        permute_467: "f32[512, 128]" = torch.ops.aten.permute.default(arg1079_1, [1, 0]);  arg1079_1 = None
        addmm_350: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1080_1, view_932, permute_467);  arg1080_1 = view_932 = permute_467 = None
        view_933: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_350, [256, 128, 128]);  addmm_350 = None
        view_934: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_933, [256, 128, -1, 32]);  view_933 = None
        permute_468: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_96: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_468, [256, 4, 128, 32]);  permute_468 = None
        clone_118: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_939: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_118, [1024, 128, 32]);  clone_118 = None
        bmm_47: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_938, view_939);  view_938 = view_939 = None
        view_940: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_47, [256, 4, 128, 32]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_470: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_940, [0, 2, 1, 3]);  view_940 = None
        clone_119: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_941: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_119, [256, 128, -1]);  clone_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_942: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_941, [32768, 128]);  view_941 = None
        permute_471: "f32[128, 128]" = torch.ops.aten.permute.default(arg1081_1, [1, 0]);  arg1081_1 = None
        addmm_351: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1082_1, view_942, permute_471);  arg1082_1 = view_942 = permute_471 = None
        view_943: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_351, [256, 128, 128]);  addmm_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_922: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_349, [32768, 512])
        permute_461: "f32[512, 128]" = torch.ops.aten.permute.default(arg1067_1, [1, 0]);  arg1067_1 = None
        addmm_346: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1068_1, view_922, permute_461);  arg1068_1 = view_922 = permute_461 = None
        view_923: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_346, [256, 128, 128]);  addmm_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_231: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_923, arg1069_1);  view_923 = arg1069_1 = None
        add_350: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_231, arg1070_1);  mul_231 = arg1070_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_353: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_943, add_350);  view_943 = add_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_235: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_353, arg1083_1);  add_353 = arg1083_1 = None
        add_354: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_235, arg1084_1);  mul_235 = arg1084_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_944: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_354, [32768, 128])
        permute_472: "f32[128, 512]" = torch.ops.aten.permute.default(arg1085_1, [1, 0]);  arg1085_1 = None
        addmm_352: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1086_1, view_944, permute_472);  arg1086_1 = view_944 = permute_472 = None
        view_945: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_352, [256, 128, 512]);  addmm_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_92: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_945);  view_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_946: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_92, [32768, 512]);  relu_92 = None
        permute_473: "f32[512, 128]" = torch.ops.aten.permute.default(arg1087_1, [1, 0]);  arg1087_1 = None
        addmm_353: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1088_1, view_946, permute_473);  arg1088_1 = view_946 = permute_473 = None
        view_947: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_353, [256, 128, 128]);  addmm_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_355: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_947, add_354);  view_947 = add_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_236: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_355, arg1089_1);  add_355 = arg1089_1 = None
        add_356: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_236, arg1090_1);  mul_236 = arg1090_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_948: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_356, [32768, 128])
        permute_474: "f32[128, 512]" = torch.ops.aten.permute.default(arg1091_1, [1, 0]);  arg1091_1 = None
        addmm_354: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1092_1, view_948, permute_474);  arg1092_1 = view_948 = permute_474 = None
        view_949: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_354, [256, 128, 512]);  addmm_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_93: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_949);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_950: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_93, [32768, 512]);  relu_93 = None
        permute_475: "f32[512, 128]" = torch.ops.aten.permute.default(arg1093_1, [1, 0]);  arg1093_1 = None
        addmm_355: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1094_1, view_950, permute_475);  arg1094_1 = view_950 = permute_475 = None
        view_951: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_355, [256, 128, 128]);  addmm_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_357: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_951, add_356);  view_951 = add_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_237: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_357, arg1095_1);  add_357 = arg1095_1 = None
        add_358: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_237, arg1096_1);  mul_237 = arg1096_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_952: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_358, [32768, 128])
        permute_476: "f32[128, 512]" = torch.ops.aten.permute.default(arg1097_1, [1, 0]);  arg1097_1 = None
        addmm_356: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1098_1, view_952, permute_476);  arg1098_1 = view_952 = permute_476 = None
        view_953: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_356, [256, 128, 512]);  addmm_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_94: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_953);  view_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_954: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_94, [32768, 512]);  relu_94 = None
        permute_477: "f32[512, 128]" = torch.ops.aten.permute.default(arg1099_1, [1, 0]);  arg1099_1 = None
        addmm_357: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1100_1, view_954, permute_477);  arg1100_1 = view_954 = permute_477 = None
        view_955: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_357, [256, 128, 128]);  addmm_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_359: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_955, add_358);  view_955 = add_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_238: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_359, arg1101_1);  add_359 = arg1101_1 = None
        add_360: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_238, arg1102_1);  mul_238 = arg1102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_956: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_360, [32768, 128])
        permute_478: "f32[128, 512]" = torch.ops.aten.permute.default(arg1103_1, [1, 0]);  arg1103_1 = None
        addmm_358: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1104_1, view_956, permute_478);  arg1104_1 = view_956 = permute_478 = None
        view_957: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_358, [256, 128, 512]);  addmm_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_95: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_957);  view_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_958: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_95, [32768, 512]);  relu_95 = None
        permute_479: "f32[512, 128]" = torch.ops.aten.permute.default(arg1105_1, [1, 0]);  arg1105_1 = None
        addmm_359: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg1106_1, view_958, permute_479);  arg1106_1 = view_958 = permute_479 = None
        view_959: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_359, [256, 128, 128]);  addmm_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_361: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_959, add_360);  view_959 = add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_239: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_361, arg1107_1);  add_361 = arg1107_1 = None
        add_362: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_239, arg1108_1);  mul_239 = arg1108_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_960: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_362, [32768, 128]);  add_362 = None
        permute_480: "f32[128, 512]" = torch.ops.aten.permute.default(arg1109_1, [1, 0]);  arg1109_1 = None
        addmm_360: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1110_1, view_960, permute_480);  arg1110_1 = view_960 = permute_480 = None
        view_961: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_360, [256, 128, 512]);  addmm_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_363: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_961, add_349);  view_961 = add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_240: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_363, arg1111_1);  add_363 = arg1111_1 = None
        add_364: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_240, arg1112_1);  mul_240 = arg1112_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:489 in forward, code: hidden_states = self.dense(hidden_states)
        view_962: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_364, [32768, 512]);  add_364 = None
        permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(arg1113_1, [1, 0]);  arg1113_1 = None
        addmm_361: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg1114_1, view_962, permute_481);  arg1114_1 = view_962 = permute_481 = None
        view_963: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_361, [256, 128, 512]);  addmm_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:490 in forward, code: hidden_states = self.transform_act_fn(hidden_states)
        relu_96: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_963);  view_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:491 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(relu_96, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean[0]
        getitem_1: "f32[256, 128, 1]" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:825 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_969: "i64[32768]" = torch.ops.aten.reshape.default(arg1119_1, [-1]);  arg1119_1 = None
        ne_1: "b8[32768]" = torch.ops.aten.ne.Scalar(view_969, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:491 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_24: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(relu_96, getitem_1);  relu_96 = getitem_1 = None
        add_365: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_365);  add_365 = None
        mul_241: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt);  sub_24 = rsqrt = None
        mul_242: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_241, arg1115_1);  mul_241 = arg1115_1 = None
        add_366: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_242, arg1116_1);  mul_242 = arg1116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        view_964: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_366, [32768, 512]);  add_366 = None
        permute_482: "f32[128, 30522]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        cat_1: "f32[512, 30522]" = torch.ops.aten.cat.default([permute_482, arg1117_1]);  permute_482 = arg1117_1 = None
        constant_pad_nd_default: "f32[512, 30524]" = torch.ops.aten.constant_pad_nd.default(cat_1, [0, 2, 0, 0]);  cat_1 = None
        mm_default: "f32[32768, 30524]" = torch.ops.aten.mm.default(view_964, constant_pad_nd_default);  view_964 = constant_pad_nd_default = None
        slice_tensor: "f32[32768, 30522]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -2);  mm_default = None
        view_965: "f32[256, 128, 30522]" = torch.ops.aten.reshape.default(slice_tensor, [256, 128, 30522]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:508 in forward, code: hidden_states += self.decoder.bias
        add_367: "f32[256, 128, 30522]" = torch.ops.aten.add.Tensor(view_965, arg1118_1);  view_965 = arg1118_1 = None
        view_966: "f32[32768, 30522]" = torch.ops.aten.reshape.default(add_367, [32768, 30522]);  add_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:825 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        amax_24: "f32[32768, 1]" = torch.ops.aten.amax.default(view_966, [1], True)
        sub_25: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(view_966, amax_24);  amax_24 = None
        exp_24: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_25)
        sum_25: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_26: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view_969, -100)
        full_default_73: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48: "i64[32768]" = torch.ops.aten.where.self(ne, view_969, full_default_73);  ne = full_default_73 = None
        unsqueeze_3: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_48, 1);  where_48 = None
        gather: "f32[32768, 1]" = torch.ops.aten.gather.default(sub_26, 1, unsqueeze_3);  sub_26 = unsqueeze_3 = None
        squeeze: "f32[32768]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_74: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_49: "f32[32768]" = torch.ops.aten.where.self(ne_1, neg, full_default_74);  ne_1 = neg = full_default_74 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_49);  where_49 = None
        ne_2: "b8[32768]" = torch.ops.aten.ne.Scalar(view_969, -100);  view_969 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:508 in forward, code: hidden_states += self.decoder.bias
        view_967: "f32[256, 128, 30522]" = torch.ops.aten.reshape.default(view_966, [256, 128, 30522]);  view_966 = None
        return (div_24, view_967)

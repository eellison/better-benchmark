class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 512][512, 1]cuda:0", arg1_1: "i64[1, 512][512, 1]cuda:0", arg2_1: "bf16[128100, 1536][1536, 1]cuda:0", arg3_1: "bf16[512, 1536][1536, 1]cuda:0", arg4_1: "bf16[1536][1]cuda:0", arg5_1: "bf16[1536][1]cuda:0", arg6_1: "bf16[1536, 1536][1536, 1]cuda:0", arg7_1: "bf16[1536][1]cuda:0", arg8_1: "bf16[1536, 1536][1536, 1]cuda:0", arg9_1: "bf16[1536][1]cuda:0", arg10_1: "bf16[1536, 1536][1536, 1]cuda:0", arg11_1: "bf16[1536][1]cuda:0", arg12_1: "bf16[1536, 1536][1536, 1]cuda:0", arg13_1: "bf16[1536][1]cuda:0", arg14_1: "bf16[1536][1]cuda:0", arg15_1: "bf16[1536][1]cuda:0", arg16_1: "bf16[6144, 1536][1536, 1]cuda:0", arg17_1: "bf16[6144][1]cuda:0", arg18_1: "bf16[1536, 6144][6144, 1]cuda:0", arg19_1: "bf16[1536][1]cuda:0", arg20_1: "bf16[1536][1]cuda:0", arg21_1: "bf16[1536][1]cuda:0", arg22_1: "bf16[1536, 1536][1536, 1]cuda:0", arg23_1: "bf16[1536][1]cuda:0", arg24_1: "bf16[1536, 1536][1536, 1]cuda:0", arg25_1: "bf16[1536][1]cuda:0", arg26_1: "bf16[1536, 1536][1536, 1]cuda:0", arg27_1: "bf16[1536][1]cuda:0", arg28_1: "bf16[1536, 1536][1536, 1]cuda:0", arg29_1: "bf16[1536][1]cuda:0", arg30_1: "bf16[1536][1]cuda:0", arg31_1: "bf16[1536][1]cuda:0", arg32_1: "bf16[6144, 1536][1536, 1]cuda:0", arg33_1: "bf16[6144][1]cuda:0", arg34_1: "bf16[1536, 6144][6144, 1]cuda:0", arg35_1: "bf16[1536][1]cuda:0", arg36_1: "bf16[1536][1]cuda:0", arg37_1: "bf16[1536][1]cuda:0", arg38_1: "bf16[1536, 1536][1536, 1]cuda:0", arg39_1: "bf16[1536][1]cuda:0", arg40_1: "bf16[1536, 1536][1536, 1]cuda:0", arg41_1: "bf16[1536][1]cuda:0", arg42_1: "bf16[1536, 1536][1536, 1]cuda:0", arg43_1: "bf16[1536][1]cuda:0", arg44_1: "bf16[1536, 1536][1536, 1]cuda:0", arg45_1: "bf16[1536][1]cuda:0", arg46_1: "bf16[1536][1]cuda:0", arg47_1: "bf16[1536][1]cuda:0", arg48_1: "bf16[6144, 1536][1536, 1]cuda:0", arg49_1: "bf16[6144][1]cuda:0", arg50_1: "bf16[1536, 6144][6144, 1]cuda:0", arg51_1: "bf16[1536][1]cuda:0", arg52_1: "bf16[1536][1]cuda:0", arg53_1: "bf16[1536][1]cuda:0", arg54_1: "bf16[1536, 1536][1536, 1]cuda:0", arg55_1: "bf16[1536][1]cuda:0", arg56_1: "bf16[1536, 1536][1536, 1]cuda:0", arg57_1: "bf16[1536][1]cuda:0", arg58_1: "bf16[1536, 1536][1536, 1]cuda:0", arg59_1: "bf16[1536][1]cuda:0", arg60_1: "bf16[1536, 1536][1536, 1]cuda:0", arg61_1: "bf16[1536][1]cuda:0", arg62_1: "bf16[1536][1]cuda:0", arg63_1: "bf16[1536][1]cuda:0", arg64_1: "bf16[6144, 1536][1536, 1]cuda:0", arg65_1: "bf16[6144][1]cuda:0", arg66_1: "bf16[1536, 6144][6144, 1]cuda:0", arg67_1: "bf16[1536][1]cuda:0", arg68_1: "bf16[1536][1]cuda:0", arg69_1: "bf16[1536][1]cuda:0", arg70_1: "bf16[1536, 1536][1536, 1]cuda:0", arg71_1: "bf16[1536][1]cuda:0", arg72_1: "bf16[1536, 1536][1536, 1]cuda:0", arg73_1: "bf16[1536][1]cuda:0", arg74_1: "bf16[1536, 1536][1536, 1]cuda:0", arg75_1: "bf16[1536][1]cuda:0", arg76_1: "bf16[1536, 1536][1536, 1]cuda:0", arg77_1: "bf16[1536][1]cuda:0", arg78_1: "bf16[1536][1]cuda:0", arg79_1: "bf16[1536][1]cuda:0", arg80_1: "bf16[6144, 1536][1536, 1]cuda:0", arg81_1: "bf16[6144][1]cuda:0", arg82_1: "bf16[1536, 6144][6144, 1]cuda:0", arg83_1: "bf16[1536][1]cuda:0", arg84_1: "bf16[1536][1]cuda:0", arg85_1: "bf16[1536][1]cuda:0", arg86_1: "bf16[1536, 1536][1536, 1]cuda:0", arg87_1: "bf16[1536][1]cuda:0", arg88_1: "bf16[1536, 1536][1536, 1]cuda:0", arg89_1: "bf16[1536][1]cuda:0", arg90_1: "bf16[1536, 1536][1536, 1]cuda:0", arg91_1: "bf16[1536][1]cuda:0", arg92_1: "bf16[1536, 1536][1536, 1]cuda:0", arg93_1: "bf16[1536][1]cuda:0", arg94_1: "bf16[1536][1]cuda:0", arg95_1: "bf16[1536][1]cuda:0", arg96_1: "bf16[6144, 1536][1536, 1]cuda:0", arg97_1: "bf16[6144][1]cuda:0", arg98_1: "bf16[1536, 6144][6144, 1]cuda:0", arg99_1: "bf16[1536][1]cuda:0", arg100_1: "bf16[1536][1]cuda:0", arg101_1: "bf16[1536][1]cuda:0", arg102_1: "bf16[1536, 1536][1536, 1]cuda:0", arg103_1: "bf16[1536][1]cuda:0", arg104_1: "bf16[1536, 1536][1536, 1]cuda:0", arg105_1: "bf16[1536][1]cuda:0", arg106_1: "bf16[1536, 1536][1536, 1]cuda:0", arg107_1: "bf16[1536][1]cuda:0", arg108_1: "bf16[1536, 1536][1536, 1]cuda:0", arg109_1: "bf16[1536][1]cuda:0", arg110_1: "bf16[1536][1]cuda:0", arg111_1: "bf16[1536][1]cuda:0", arg112_1: "bf16[6144, 1536][1536, 1]cuda:0", arg113_1: "bf16[6144][1]cuda:0", arg114_1: "bf16[1536, 6144][6144, 1]cuda:0", arg115_1: "bf16[1536][1]cuda:0", arg116_1: "bf16[1536][1]cuda:0", arg117_1: "bf16[1536][1]cuda:0", arg118_1: "bf16[1536, 1536][1536, 1]cuda:0", arg119_1: "bf16[1536][1]cuda:0", arg120_1: "bf16[1536, 1536][1536, 1]cuda:0", arg121_1: "bf16[1536][1]cuda:0", arg122_1: "bf16[1536, 1536][1536, 1]cuda:0", arg123_1: "bf16[1536][1]cuda:0", arg124_1: "bf16[1536, 1536][1536, 1]cuda:0", arg125_1: "bf16[1536][1]cuda:0", arg126_1: "bf16[1536][1]cuda:0", arg127_1: "bf16[1536][1]cuda:0", arg128_1: "bf16[6144, 1536][1536, 1]cuda:0", arg129_1: "bf16[6144][1]cuda:0", arg130_1: "bf16[1536, 6144][6144, 1]cuda:0", arg131_1: "bf16[1536][1]cuda:0", arg132_1: "bf16[1536][1]cuda:0", arg133_1: "bf16[1536][1]cuda:0", arg134_1: "bf16[1536, 1536][1536, 1]cuda:0", arg135_1: "bf16[1536][1]cuda:0", arg136_1: "bf16[1536, 1536][1536, 1]cuda:0", arg137_1: "bf16[1536][1]cuda:0", arg138_1: "bf16[1536, 1536][1536, 1]cuda:0", arg139_1: "bf16[1536][1]cuda:0", arg140_1: "bf16[1536, 1536][1536, 1]cuda:0", arg141_1: "bf16[1536][1]cuda:0", arg142_1: "bf16[1536][1]cuda:0", arg143_1: "bf16[1536][1]cuda:0", arg144_1: "bf16[6144, 1536][1536, 1]cuda:0", arg145_1: "bf16[6144][1]cuda:0", arg146_1: "bf16[1536, 6144][6144, 1]cuda:0", arg147_1: "bf16[1536][1]cuda:0", arg148_1: "bf16[1536][1]cuda:0", arg149_1: "bf16[1536][1]cuda:0", arg150_1: "bf16[1536, 1536][1536, 1]cuda:0", arg151_1: "bf16[1536][1]cuda:0", arg152_1: "bf16[1536, 1536][1536, 1]cuda:0", arg153_1: "bf16[1536][1]cuda:0", arg154_1: "bf16[1536, 1536][1536, 1]cuda:0", arg155_1: "bf16[1536][1]cuda:0", arg156_1: "bf16[1536, 1536][1536, 1]cuda:0", arg157_1: "bf16[1536][1]cuda:0", arg158_1: "bf16[1536][1]cuda:0", arg159_1: "bf16[1536][1]cuda:0", arg160_1: "bf16[6144, 1536][1536, 1]cuda:0", arg161_1: "bf16[6144][1]cuda:0", arg162_1: "bf16[1536, 6144][6144, 1]cuda:0", arg163_1: "bf16[1536][1]cuda:0", arg164_1: "bf16[1536][1]cuda:0", arg165_1: "bf16[1536][1]cuda:0", arg166_1: "bf16[1536, 1536][1536, 1]cuda:0", arg167_1: "bf16[1536][1]cuda:0", arg168_1: "bf16[1536, 1536][1536, 1]cuda:0", arg169_1: "bf16[1536][1]cuda:0", arg170_1: "bf16[1536, 1536][1536, 1]cuda:0", arg171_1: "bf16[1536][1]cuda:0", arg172_1: "bf16[1536, 1536][1536, 1]cuda:0", arg173_1: "bf16[1536][1]cuda:0", arg174_1: "bf16[1536][1]cuda:0", arg175_1: "bf16[1536][1]cuda:0", arg176_1: "bf16[6144, 1536][1536, 1]cuda:0", arg177_1: "bf16[6144][1]cuda:0", arg178_1: "bf16[1536, 6144][6144, 1]cuda:0", arg179_1: "bf16[1536][1]cuda:0", arg180_1: "bf16[1536][1]cuda:0", arg181_1: "bf16[1536][1]cuda:0", arg182_1: "bf16[1536, 1536][1536, 1]cuda:0", arg183_1: "bf16[1536][1]cuda:0", arg184_1: "bf16[1536, 1536][1536, 1]cuda:0", arg185_1: "bf16[1536][1]cuda:0", arg186_1: "bf16[1536, 1536][1536, 1]cuda:0", arg187_1: "bf16[1536][1]cuda:0", arg188_1: "bf16[1536, 1536][1536, 1]cuda:0", arg189_1: "bf16[1536][1]cuda:0", arg190_1: "bf16[1536][1]cuda:0", arg191_1: "bf16[1536][1]cuda:0", arg192_1: "bf16[6144, 1536][1536, 1]cuda:0", arg193_1: "bf16[6144][1]cuda:0", arg194_1: "bf16[1536, 6144][6144, 1]cuda:0", arg195_1: "bf16[1536][1]cuda:0", arg196_1: "bf16[1536][1]cuda:0", arg197_1: "bf16[1536][1]cuda:0", arg198_1: "bf16[1536, 1536][1536, 1]cuda:0", arg199_1: "bf16[1536][1]cuda:0", arg200_1: "bf16[1536, 1536][1536, 1]cuda:0", arg201_1: "bf16[1536][1]cuda:0", arg202_1: "bf16[1536, 1536][1536, 1]cuda:0", arg203_1: "bf16[1536][1]cuda:0", arg204_1: "bf16[1536, 1536][1536, 1]cuda:0", arg205_1: "bf16[1536][1]cuda:0", arg206_1: "bf16[1536][1]cuda:0", arg207_1: "bf16[1536][1]cuda:0", arg208_1: "bf16[6144, 1536][1536, 1]cuda:0", arg209_1: "bf16[6144][1]cuda:0", arg210_1: "bf16[1536, 6144][6144, 1]cuda:0", arg211_1: "bf16[1536][1]cuda:0", arg212_1: "bf16[1536][1]cuda:0", arg213_1: "bf16[1536][1]cuda:0", arg214_1: "bf16[1536, 1536][1536, 1]cuda:0", arg215_1: "bf16[1536][1]cuda:0", arg216_1: "bf16[1536, 1536][1536, 1]cuda:0", arg217_1: "bf16[1536][1]cuda:0", arg218_1: "bf16[1536, 1536][1536, 1]cuda:0", arg219_1: "bf16[1536][1]cuda:0", arg220_1: "bf16[1536, 1536][1536, 1]cuda:0", arg221_1: "bf16[1536][1]cuda:0", arg222_1: "bf16[1536][1]cuda:0", arg223_1: "bf16[1536][1]cuda:0", arg224_1: "bf16[6144, 1536][1536, 1]cuda:0", arg225_1: "bf16[6144][1]cuda:0", arg226_1: "bf16[1536, 6144][6144, 1]cuda:0", arg227_1: "bf16[1536][1]cuda:0", arg228_1: "bf16[1536][1]cuda:0", arg229_1: "bf16[1536][1]cuda:0", arg230_1: "bf16[1536, 1536][1536, 1]cuda:0", arg231_1: "bf16[1536][1]cuda:0", arg232_1: "bf16[1536, 1536][1536, 1]cuda:0", arg233_1: "bf16[1536][1]cuda:0", arg234_1: "bf16[1536, 1536][1536, 1]cuda:0", arg235_1: "bf16[1536][1]cuda:0", arg236_1: "bf16[1536, 1536][1536, 1]cuda:0", arg237_1: "bf16[1536][1]cuda:0", arg238_1: "bf16[1536][1]cuda:0", arg239_1: "bf16[1536][1]cuda:0", arg240_1: "bf16[6144, 1536][1536, 1]cuda:0", arg241_1: "bf16[6144][1]cuda:0", arg242_1: "bf16[1536, 6144][6144, 1]cuda:0", arg243_1: "bf16[1536][1]cuda:0", arg244_1: "bf16[1536][1]cuda:0", arg245_1: "bf16[1536][1]cuda:0", arg246_1: "bf16[1536, 1536][1536, 1]cuda:0", arg247_1: "bf16[1536][1]cuda:0", arg248_1: "bf16[1536, 1536][1536, 1]cuda:0", arg249_1: "bf16[1536][1]cuda:0", arg250_1: "bf16[1536, 1536][1536, 1]cuda:0", arg251_1: "bf16[1536][1]cuda:0", arg252_1: "bf16[1536, 1536][1536, 1]cuda:0", arg253_1: "bf16[1536][1]cuda:0", arg254_1: "bf16[1536][1]cuda:0", arg255_1: "bf16[1536][1]cuda:0", arg256_1: "bf16[6144, 1536][1536, 1]cuda:0", arg257_1: "bf16[6144][1]cuda:0", arg258_1: "bf16[1536, 6144][6144, 1]cuda:0", arg259_1: "bf16[1536][1]cuda:0", arg260_1: "bf16[1536][1]cuda:0", arg261_1: "bf16[1536][1]cuda:0", arg262_1: "bf16[1536, 1536][1536, 1]cuda:0", arg263_1: "bf16[1536][1]cuda:0", arg264_1: "bf16[1536, 1536][1536, 1]cuda:0", arg265_1: "bf16[1536][1]cuda:0", arg266_1: "bf16[1536, 1536][1536, 1]cuda:0", arg267_1: "bf16[1536][1]cuda:0", arg268_1: "bf16[1536, 1536][1536, 1]cuda:0", arg269_1: "bf16[1536][1]cuda:0", arg270_1: "bf16[1536][1]cuda:0", arg271_1: "bf16[1536][1]cuda:0", arg272_1: "bf16[6144, 1536][1536, 1]cuda:0", arg273_1: "bf16[6144][1]cuda:0", arg274_1: "bf16[1536, 6144][6144, 1]cuda:0", arg275_1: "bf16[1536][1]cuda:0", arg276_1: "bf16[1536][1]cuda:0", arg277_1: "bf16[1536][1]cuda:0", arg278_1: "bf16[1536, 1536][1536, 1]cuda:0", arg279_1: "bf16[1536][1]cuda:0", arg280_1: "bf16[1536, 1536][1536, 1]cuda:0", arg281_1: "bf16[1536][1]cuda:0", arg282_1: "bf16[1536, 1536][1536, 1]cuda:0", arg283_1: "bf16[1536][1]cuda:0", arg284_1: "bf16[1536, 1536][1536, 1]cuda:0", arg285_1: "bf16[1536][1]cuda:0", arg286_1: "bf16[1536][1]cuda:0", arg287_1: "bf16[1536][1]cuda:0", arg288_1: "bf16[6144, 1536][1536, 1]cuda:0", arg289_1: "bf16[6144][1]cuda:0", arg290_1: "bf16[1536, 6144][6144, 1]cuda:0", arg291_1: "bf16[1536][1]cuda:0", arg292_1: "bf16[1536][1]cuda:0", arg293_1: "bf16[1536][1]cuda:0", arg294_1: "bf16[1536, 1536][1536, 1]cuda:0", arg295_1: "bf16[1536][1]cuda:0", arg296_1: "bf16[1536, 1536][1536, 1]cuda:0", arg297_1: "bf16[1536][1]cuda:0", arg298_1: "bf16[1536, 1536][1536, 1]cuda:0", arg299_1: "bf16[1536][1]cuda:0", arg300_1: "bf16[1536, 1536][1536, 1]cuda:0", arg301_1: "bf16[1536][1]cuda:0", arg302_1: "bf16[1536][1]cuda:0", arg303_1: "bf16[1536][1]cuda:0", arg304_1: "bf16[6144, 1536][1536, 1]cuda:0", arg305_1: "bf16[6144][1]cuda:0", arg306_1: "bf16[1536, 6144][6144, 1]cuda:0", arg307_1: "bf16[1536][1]cuda:0", arg308_1: "bf16[1536][1]cuda:0", arg309_1: "bf16[1536][1]cuda:0", arg310_1: "bf16[1536, 1536][1536, 1]cuda:0", arg311_1: "bf16[1536][1]cuda:0", arg312_1: "bf16[1536, 1536][1536, 1]cuda:0", arg313_1: "bf16[1536][1]cuda:0", arg314_1: "bf16[1536, 1536][1536, 1]cuda:0", arg315_1: "bf16[1536][1]cuda:0", arg316_1: "bf16[1536, 1536][1536, 1]cuda:0", arg317_1: "bf16[1536][1]cuda:0", arg318_1: "bf16[1536][1]cuda:0", arg319_1: "bf16[1536][1]cuda:0", arg320_1: "bf16[6144, 1536][1536, 1]cuda:0", arg321_1: "bf16[6144][1]cuda:0", arg322_1: "bf16[1536, 6144][6144, 1]cuda:0", arg323_1: "bf16[1536][1]cuda:0", arg324_1: "bf16[1536][1]cuda:0", arg325_1: "bf16[1536][1]cuda:0", arg326_1: "bf16[1536, 1536][1536, 1]cuda:0", arg327_1: "bf16[1536][1]cuda:0", arg328_1: "bf16[1536, 1536][1536, 1]cuda:0", arg329_1: "bf16[1536][1]cuda:0", arg330_1: "bf16[1536, 1536][1536, 1]cuda:0", arg331_1: "bf16[1536][1]cuda:0", arg332_1: "bf16[1536, 1536][1536, 1]cuda:0", arg333_1: "bf16[1536][1]cuda:0", arg334_1: "bf16[1536][1]cuda:0", arg335_1: "bf16[1536][1]cuda:0", arg336_1: "bf16[6144, 1536][1536, 1]cuda:0", arg337_1: "bf16[6144][1]cuda:0", arg338_1: "bf16[1536, 6144][6144, 1]cuda:0", arg339_1: "bf16[1536][1]cuda:0", arg340_1: "bf16[1536][1]cuda:0", arg341_1: "bf16[1536][1]cuda:0", arg342_1: "bf16[1536, 1536][1536, 1]cuda:0", arg343_1: "bf16[1536][1]cuda:0", arg344_1: "bf16[1536, 1536][1536, 1]cuda:0", arg345_1: "bf16[1536][1]cuda:0", arg346_1: "bf16[1536, 1536][1536, 1]cuda:0", arg347_1: "bf16[1536][1]cuda:0", arg348_1: "bf16[1536, 1536][1536, 1]cuda:0", arg349_1: "bf16[1536][1]cuda:0", arg350_1: "bf16[1536][1]cuda:0", arg351_1: "bf16[1536][1]cuda:0", arg352_1: "bf16[6144, 1536][1536, 1]cuda:0", arg353_1: "bf16[6144][1]cuda:0", arg354_1: "bf16[1536, 6144][6144, 1]cuda:0", arg355_1: "bf16[1536][1]cuda:0", arg356_1: "bf16[1536][1]cuda:0", arg357_1: "bf16[1536][1]cuda:0", arg358_1: "bf16[1536, 1536][1536, 1]cuda:0", arg359_1: "bf16[1536][1]cuda:0", arg360_1: "bf16[1536, 1536][1536, 1]cuda:0", arg361_1: "bf16[1536][1]cuda:0", arg362_1: "bf16[1536, 1536][1536, 1]cuda:0", arg363_1: "bf16[1536][1]cuda:0", arg364_1: "bf16[1536, 1536][1536, 1]cuda:0", arg365_1: "bf16[1536][1]cuda:0", arg366_1: "bf16[1536][1]cuda:0", arg367_1: "bf16[1536][1]cuda:0", arg368_1: "bf16[6144, 1536][1536, 1]cuda:0", arg369_1: "bf16[6144][1]cuda:0", arg370_1: "bf16[1536, 6144][6144, 1]cuda:0", arg371_1: "bf16[1536][1]cuda:0", arg372_1: "bf16[1536][1]cuda:0", arg373_1: "bf16[1536][1]cuda:0", arg374_1: "bf16[1536, 1536][1536, 1]cuda:0", arg375_1: "bf16[1536][1]cuda:0", arg376_1: "bf16[1536, 1536][1536, 1]cuda:0", arg377_1: "bf16[1536][1]cuda:0", arg378_1: "bf16[1536, 1536][1536, 1]cuda:0", arg379_1: "bf16[1536][1]cuda:0", arg380_1: "bf16[1536, 1536][1536, 1]cuda:0", arg381_1: "bf16[1536][1]cuda:0", arg382_1: "bf16[1536][1]cuda:0", arg383_1: "bf16[1536][1]cuda:0", arg384_1: "bf16[6144, 1536][1536, 1]cuda:0", arg385_1: "bf16[6144][1]cuda:0", arg386_1: "bf16[1536, 6144][6144, 1]cuda:0", arg387_1: "bf16[1536][1]cuda:0", arg388_1: "bf16[1536][1]cuda:0", arg389_1: "bf16[1536][1]cuda:0", arg390_1: "bf16[1536, 1536][1536, 1]cuda:0", arg391_1: "bf16[1536][1]cuda:0", arg392_1: "bf16[1536][1]cuda:0", arg393_1: "bf16[1536][1]cuda:0", arg394_1: "bf16[128100][1]cuda:0", arg395_1: "i64[8, 512][512, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:535 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:538 in forward, code: position_embeddings = self.position_embeddings(position_ids.long())
        embedding_1: "bf16[1, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.embedding.default(arg3_1, arg1_1);  arg3_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:544 in forward, code: embeddings = embeddings + position_embeddings
        add: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        convert_element_type: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:753 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full: "f32[8, 512][512, 1]cuda:0" = torch.ops.aten.full.default([8, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:558 in forward, code: mask = mask.unsqueeze(2)
        unsqueeze: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(full, 2);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:559 in forward, code: mask = mask.to(embeddings.dtype)
        full_default: "bf16[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 512, 1], 1.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant0: "f32[][]cpu" = self._tensor_constant0
        lift_fresh_copy: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        mul_4: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy, 1);  lift_fresh_copy = None
        sqrt: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_4);  mul_4 = sqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:605 in get_attention_mask, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        unsqueeze_1: "f32[8, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_2: "f32[8, 1, 1, 512][512, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:606 in get_attention_mask, code: attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        squeeze: "f32[8, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(unsqueeze_2, -2)
        unsqueeze_3: "f32[8, 1, 512, 1][512, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
        mul_3: "f32[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_15: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_2: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        sub: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
        rsqrt: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:561 in forward, code: embeddings = embeddings * mask
        convert_element_type_1: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 1536])
        permute: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [8, 512, 1536]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_2: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [8, 512, 24, -1]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_1: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone_1: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [-1, 512, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_4: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 1536])
        permute_2: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg9_1, view_4, permute_2);  arg9_1 = view_4 = permute_2 = None
        view_5: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [8, 512, 1536]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_6: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [8, 512, 24, -1]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_3: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_2: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_7: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [-1, 512, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_6: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        full_default_1: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_6, full_default_1);  permute_6 = full_default_1 = None
        bmm: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_3, div);  view_3 = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_12: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [-1, 24, 512, 512]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_3, view_12);  full_default_2 = full_default_3 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_16: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_16, [-1], True)
        sub_1: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_16, amax);  convert_element_type_16 = amax = None
        exp: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_17: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_13: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_17, [-1, 512, 512]);  convert_element_type_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_8: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 1536])
        permute_4: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg11_1, view_8, permute_4);  arg11_1 = view_8 = permute_4 = None
        view_9: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [8, 512, 1536]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_10: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [8, 512, 24, -1]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_5: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [-1, 512, 64]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_1: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_13, view_11);  view_13 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_14: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [-1, 24, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_7: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_5: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_15: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [8, 512, -1]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [4096, 1536]);  view_15 = None
        permute_8: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg13_1, view_16, permute_8);  arg13_1 = view_16 = permute_8 = None
        view_17: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [8, 512, 1536]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_3: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_17, convert_element_type_1);  view_17 = convert_element_type_1 = None
        convert_element_type_23: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_23, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, getitem_3);  convert_element_type_23 = getitem_3 = None
        add_4: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-07);  getitem_2 = None
        rsqrt_1: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_5: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        add_5: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, arg15_1);  mul_6 = arg15_1 = None
        convert_element_type_24: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_24, [4096, 1536])
        permute_9: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg17_1, view_18, permute_9);  arg17_1 = view_18 = permute_9 = None
        view_19: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [8, 512, 6144]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_28: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_7: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.5)
        mul_8: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.7071067811865476);  convert_element_type_28 = None
        erf: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_6: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, add_6);  mul_7 = add_6 = None
        convert_element_type_29: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_29, [4096, 6144]);  convert_element_type_29 = None
        permute_10: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg19_1, view_20, permute_10);  arg19_1 = view_20 = permute_10 = None
        view_21: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [8, 512, 1536]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_7: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_21, convert_element_type_24);  view_21 = convert_element_type_24 = None
        convert_element_type_33: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant1: "f32[][]cpu" = self._tensor_constant1
        lift_fresh_copy_1: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        mul_12: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_1, 1);  lift_fresh_copy_1 = None
        sqrt_1: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_12);  mul_12 = sqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_47: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_5: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_3: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_5);  convert_element_type_33 = getitem_5 = None
        add_8: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-07);  getitem_4 = None
        rsqrt_2: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_10: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_11: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        add_9: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, arg21_1);  mul_11 = arg21_1 = None
        convert_element_type_34: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_22: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [4096, 1536])
        permute_11: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg23_1, view_22, permute_11);  arg23_1 = view_22 = permute_11 = None
        view_23: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [8, 512, 1536]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_24: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [8, 512, 24, -1]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_12: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_8: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_25: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [-1, 512, 64]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_26: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [4096, 1536])
        permute_13: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg25_1, view_26, permute_13);  arg25_1 = view_26 = permute_13 = None
        view_27: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [8, 512, 1536]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_28: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_27, [8, 512, 24, -1]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_14: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_9: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_29: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [-1, 512, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_17: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
        full_default_4: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_2: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_17, full_default_4);  permute_17 = full_default_4 = None
        bmm_2: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_25, div_2);  view_25 = div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_34: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [-1, 24, 512, 512]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_1: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_5, full_default_6, view_34);  full_default_5 = full_default_6 = view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_48: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        amax_1: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_48, [-1], True)
        sub_4: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_48, amax_1);  convert_element_type_48 = amax_1 = None
        exp_1: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_49: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_35: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [-1, 512, 512]);  convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_30: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [4096, 1536])
        permute_15: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg27_1, view_30, permute_15);  arg27_1 = view_30 = permute_15 = None
        view_31: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [8, 512, 1536]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_32: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_31, [8, 512, 24, -1]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_16: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_10: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_33: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [-1, 512, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_3: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_35, view_33);  view_35 = view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_36: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [-1, 24, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_18: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_12: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_37: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [8, 512, -1]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [4096, 1536]);  view_37 = None
        permute_19: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg29_1, view_38, permute_19);  arg29_1 = view_38 = permute_19 = None
        view_39: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [8, 512, 1536]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_10: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_39, convert_element_type_34);  view_39 = convert_element_type_34 = None
        convert_element_type_55: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_55, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_55, getitem_7);  convert_element_type_55 = getitem_7 = None
        add_11: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-07);  getitem_6 = None
        rsqrt_3: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_13: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_14: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        add_12: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_14, arg31_1);  mul_14 = arg31_1 = None
        convert_element_type_56: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_56, [4096, 1536])
        permute_20: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg33_1, view_40, permute_20);  arg33_1 = view_40 = permute_20 = None
        view_41: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [8, 512, 6144]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_60: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_15: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.5)
        mul_16: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.7071067811865476);  convert_element_type_60 = None
        erf_1: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_13: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_17: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, add_13);  mul_15 = add_13 = None
        convert_element_type_61: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [4096, 6144]);  convert_element_type_61 = None
        permute_21: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg35_1, view_42, permute_21);  arg35_1 = view_42 = permute_21 = None
        view_43: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [8, 512, 1536]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_14: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_43, convert_element_type_56);  view_43 = convert_element_type_56 = None
        convert_element_type_65: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_65, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant2: "f32[][]cpu" = self._tensor_constant2
        lift_fresh_copy_2: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        mul_20: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_2, 1);  lift_fresh_copy_2 = None
        sqrt_2: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_20);  mul_20 = sqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_79: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_8: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_6: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_65, getitem_9);  convert_element_type_65 = getitem_9 = None
        add_15: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-07);  getitem_8 = None
        rsqrt_4: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_18: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_19: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        add_16: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, arg37_1);  mul_19 = arg37_1 = None
        convert_element_type_66: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_44: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_66, [4096, 1536])
        permute_22: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg39_1, view_44, permute_22);  arg39_1 = view_44 = permute_22 = None
        view_45: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [8, 512, 1536]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_46: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [8, 512, 24, -1]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_23: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        clone_15: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_47: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [-1, 512, 64]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_48: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_66, [4096, 1536])
        permute_24: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg41_1, view_48, permute_24);  arg41_1 = view_48 = permute_24 = None
        view_49: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [8, 512, 1536]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_50: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_49, [8, 512, 24, -1]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_25: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_16: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_51: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [-1, 512, 64]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_28: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        full_default_7: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_4: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_28, full_default_7);  permute_28 = full_default_7 = None
        bmm_4: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_47, div_4);  view_47 = div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_56: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [-1, 24, 512, 512]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_2: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_8, full_default_9, view_56);  full_default_8 = full_default_9 = view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_80: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        amax_2: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_80, [-1], True)
        sub_7: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, amax_2);  convert_element_type_80 = amax_2 = None
        exp_2: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_81: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_57: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_81, [-1, 512, 512]);  convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_52: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_66, [4096, 1536])
        permute_26: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg43_1, view_52, permute_26);  arg43_1 = view_52 = permute_26 = None
        view_53: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [8, 512, 1536]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_54: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_53, [8, 512, 24, -1]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_27: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_17: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_55: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [-1, 512, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_5: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_57, view_55);  view_57 = view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_58: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [-1, 24, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_29: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_19: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_59: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [8, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [4096, 1536]);  view_59 = None
        permute_30: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg45_1, view_60, permute_30);  arg45_1 = view_60 = permute_30 = None
        view_61: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [8, 512, 1536]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_17: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_61, convert_element_type_66);  view_61 = convert_element_type_66 = None
        convert_element_type_87: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.float32);  add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_87, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_87, getitem_11);  convert_element_type_87 = getitem_11 = None
        add_18: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-07);  getitem_10 = None
        rsqrt_5: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_21: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_22: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
        add_19: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, arg47_1);  mul_22 = arg47_1 = None
        convert_element_type_88: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_88, [4096, 1536])
        permute_31: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg49_1, view_62, permute_31);  arg49_1 = view_62 = permute_31 = None
        view_63: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [8, 512, 6144]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_92: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_23: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.5)
        mul_24: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.7071067811865476);  convert_element_type_92 = None
        erf_2: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_20: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_25: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, add_20);  mul_23 = add_20 = None
        convert_element_type_93: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_93, [4096, 6144]);  convert_element_type_93 = None
        permute_32: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg51_1, view_64, permute_32);  arg51_1 = view_64 = permute_32 = None
        view_65: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [8, 512, 1536]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_21: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_65, convert_element_type_88);  view_65 = convert_element_type_88 = None
        convert_element_type_97: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.float32);  add_21 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_97, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant3: "f32[][]cpu" = self._tensor_constant3
        lift_fresh_copy_3: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        mul_28: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_3, 1);  lift_fresh_copy_3 = None
        sqrt_3: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_28);  mul_28 = sqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_111: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_11: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_9: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_97, getitem_13);  convert_element_type_97 = getitem_13 = None
        add_22: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-07);  getitem_12 = None
        rsqrt_6: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_26: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_27: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
        add_23: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, arg53_1);  mul_27 = arg53_1 = None
        convert_element_type_98: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_66: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_98, [4096, 1536])
        permute_33: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg55_1, view_66, permute_33);  arg55_1 = view_66 = permute_33 = None
        view_67: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [8, 512, 1536]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_68: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [8, 512, 24, -1]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_34: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_22: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_69: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [-1, 512, 64]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_70: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_98, [4096, 1536])
        permute_35: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg57_1, view_70, permute_35);  arg57_1 = view_70 = permute_35 = None
        view_71: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [8, 512, 1536]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_72: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_71, [8, 512, 24, -1]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_36: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_23: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_73: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [-1, 512, 64]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_39: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
        full_default_10: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_6: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_39, full_default_10);  permute_39 = full_default_10 = None
        bmm_6: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_69, div_6);  view_69 = div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_78: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [-1, 24, 512, 512]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_3: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_11, full_default_12, view_78);  full_default_11 = full_default_12 = view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_112: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        amax_3: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_112, [-1], True)
        sub_10: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_112, amax_3);  convert_element_type_112 = amax_3 = None
        exp_3: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_113: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_79: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_113, [-1, 512, 512]);  convert_element_type_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_74: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_98, [4096, 1536])
        permute_37: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg59_1, view_74, permute_37);  arg59_1 = view_74 = permute_37 = None
        view_75: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [8, 512, 1536]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_76: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_75, [8, 512, 24, -1]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_38: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        clone_24: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_77: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [-1, 512, 64]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_7: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_79, view_77);  view_79 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_80: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [-1, 24, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_40: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_26: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_81: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [8, 512, -1]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [4096, 1536]);  view_81 = None
        permute_41: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg61_1, view_82, permute_41);  arg61_1 = view_82 = permute_41 = None
        view_83: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [8, 512, 1536]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_24: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_83, convert_element_type_98);  view_83 = convert_element_type_98 = None
        convert_element_type_119: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.float32);  add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_119, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_119, getitem_15);  convert_element_type_119 = getitem_15 = None
        add_25: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-07);  getitem_14 = None
        rsqrt_7: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_29: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_30: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
        add_26: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, arg63_1);  mul_30 = arg63_1 = None
        convert_element_type_120: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_120, [4096, 1536])
        permute_42: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg65_1, view_84, permute_42);  arg65_1 = view_84 = permute_42 = None
        view_85: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [8, 512, 6144]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_124: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_31: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_124, 0.5)
        mul_32: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_124, 0.7071067811865476);  convert_element_type_124 = None
        erf_3: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_32);  mul_32 = None
        add_27: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_33: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, add_27);  mul_31 = add_27 = None
        convert_element_type_125: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_125, [4096, 6144]);  convert_element_type_125 = None
        permute_43: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg67_1, view_86, permute_43);  arg67_1 = view_86 = permute_43 = None
        view_87: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [8, 512, 1536]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_28: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_87, convert_element_type_120);  view_87 = convert_element_type_120 = None
        convert_element_type_129: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.float32);  add_28 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_129, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant4: "f32[][]cpu" = self._tensor_constant4
        lift_fresh_copy_4: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        mul_36: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_4, 1);  lift_fresh_copy_4 = None
        sqrt_4: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_36);  mul_36 = sqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_143: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_14: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_12: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_129, getitem_17);  convert_element_type_129 = getitem_17 = None
        add_29: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-07);  getitem_16 = None
        rsqrt_8: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_34: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_35: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
        add_30: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_35, arg69_1);  mul_35 = arg69_1 = None
        convert_element_type_130: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_88: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_130, [4096, 1536])
        permute_44: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg71_1, view_88, permute_44);  arg71_1 = view_88 = permute_44 = None
        view_89: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [8, 512, 1536]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_90: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [8, 512, 24, -1]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_45: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_29: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_91: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [-1, 512, 64]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_92: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_130, [4096, 1536])
        permute_46: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg73_1, view_92, permute_46);  arg73_1 = view_92 = permute_46 = None
        view_93: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [8, 512, 1536]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_94: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_93, [8, 512, 24, -1]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_47: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        clone_30: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_95: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [-1, 512, 64]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_50: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
        full_default_13: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_8: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_50, full_default_13);  permute_50 = full_default_13 = None
        bmm_8: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_91, div_8);  view_91 = div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_100: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [-1, 24, 512, 512]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_4: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_14, full_default_15, view_100);  full_default_14 = full_default_15 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_144: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        amax_4: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_144, [-1], True)
        sub_13: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, amax_4);  convert_element_type_144 = amax_4 = None
        exp_4: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_145: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_101: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [-1, 512, 512]);  convert_element_type_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_96: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_130, [4096, 1536])
        permute_48: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg75_1, view_96, permute_48);  arg75_1 = view_96 = permute_48 = None
        view_97: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [8, 512, 1536]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_98: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_97, [8, 512, 24, -1]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_49: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_31: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_99: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [-1, 512, 64]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_9: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_101, view_99);  view_101 = view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_102: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [-1, 24, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_51: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_33: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_103: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [8, 512, -1]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [4096, 1536]);  view_103 = None
        permute_52: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg77_1, view_104, permute_52);  arg77_1 = view_104 = permute_52 = None
        view_105: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [8, 512, 1536]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_105, convert_element_type_130);  view_105 = convert_element_type_130 = None
        convert_element_type_151: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_151, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_151, getitem_19);  convert_element_type_151 = getitem_19 = None
        add_32: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-07);  getitem_18 = None
        rsqrt_9: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_37: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_38: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
        add_33: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_38, arg79_1);  mul_38 = arg79_1 = None
        convert_element_type_152: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_152, [4096, 1536])
        permute_53: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg81_1, view_106, permute_53);  arg81_1 = view_106 = permute_53 = None
        view_107: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [8, 512, 6144]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_156: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_39: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_156, 0.5)
        mul_40: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_156, 0.7071067811865476);  convert_element_type_156 = None
        erf_4: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_34: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_41: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, add_34);  mul_39 = add_34 = None
        convert_element_type_157: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_157, [4096, 6144]);  convert_element_type_157 = None
        permute_54: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg83_1, view_108, permute_54);  arg83_1 = view_108 = permute_54 = None
        view_109: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [8, 512, 1536]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_109, convert_element_type_152);  view_109 = convert_element_type_152 = None
        convert_element_type_161: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_161, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant5: "f32[][]cpu" = self._tensor_constant5
        lift_fresh_copy_5: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        mul_44: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_5, 1);  lift_fresh_copy_5 = None
        sqrt_5: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_44);  mul_44 = sqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_175: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_17: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_15: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_161, getitem_21);  convert_element_type_161 = getitem_21 = None
        add_36: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-07);  getitem_20 = None
        rsqrt_10: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_42: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_43: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
        add_37: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, arg85_1);  mul_43 = arg85_1 = None
        convert_element_type_162: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_110: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_162, [4096, 1536])
        permute_55: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg87_1, view_110, permute_55);  arg87_1 = view_110 = permute_55 = None
        view_111: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [8, 512, 1536]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_112: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [8, 512, 24, -1]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_56: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        clone_36: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_113: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [-1, 512, 64]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_114: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_162, [4096, 1536])
        permute_57: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg89_1, view_114, permute_57);  arg89_1 = view_114 = permute_57 = None
        view_115: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [8, 512, 1536]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_116: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_115, [8, 512, 24, -1]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_58: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        clone_37: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_117: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [-1, 512, 64]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_61: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
        full_default_16: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_10: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_61, full_default_16);  permute_61 = full_default_16 = None
        bmm_10: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_113, div_10);  view_113 = div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_122: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [-1, 24, 512, 512]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_5: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_17, full_default_18, view_122);  full_default_17 = full_default_18 = view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_176: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        amax_5: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_176, [-1], True)
        sub_16: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_176, amax_5);  convert_element_type_176 = amax_5 = None
        exp_5: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_177: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_123: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_177, [-1, 512, 512]);  convert_element_type_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_118: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_162, [4096, 1536])
        permute_59: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg91_1, view_118, permute_59);  arg91_1 = view_118 = permute_59 = None
        view_119: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [8, 512, 1536]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_120: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_119, [8, 512, 24, -1]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_60: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        clone_38: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_121: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [-1, 512, 64]);  clone_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_11: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_123, view_121);  view_123 = view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_124: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [-1, 24, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_62: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_40: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_125: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [8, 512, -1]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [4096, 1536]);  view_125 = None
        permute_63: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg93_1, view_126, permute_63);  arg93_1 = view_126 = permute_63 = None
        view_127: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [8, 512, 1536]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_38: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_127, convert_element_type_162);  view_127 = convert_element_type_162 = None
        convert_element_type_183: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_183, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_183, getitem_23);  convert_element_type_183 = getitem_23 = None
        add_39: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-07);  getitem_22 = None
        rsqrt_11: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_45: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_46: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
        add_40: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, arg95_1);  mul_46 = arg95_1 = None
        convert_element_type_184: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_184, [4096, 1536])
        permute_64: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg97_1, view_128, permute_64);  arg97_1 = view_128 = permute_64 = None
        view_129: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [8, 512, 6144]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_188: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_47: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, 0.5)
        mul_48: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, 0.7071067811865476);  convert_element_type_188 = None
        erf_5: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_41: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_49: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, add_41);  mul_47 = add_41 = None
        convert_element_type_189: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_189, [4096, 6144]);  convert_element_type_189 = None
        permute_65: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg99_1, view_130, permute_65);  arg99_1 = view_130 = permute_65 = None
        view_131: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [8, 512, 1536]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_42: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_131, convert_element_type_184);  view_131 = convert_element_type_184 = None
        convert_element_type_193: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_42, torch.float32);  add_42 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_193, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant6: "f32[][]cpu" = self._tensor_constant6
        lift_fresh_copy_6: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        mul_52: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_6, 1);  lift_fresh_copy_6 = None
        sqrt_6: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_52);  mul_52 = sqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_207: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_20: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_18: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_193, getitem_25);  convert_element_type_193 = getitem_25 = None
        add_43: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-07);  getitem_24 = None
        rsqrt_12: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_50: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_51: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
        add_44: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, arg101_1);  mul_51 = arg101_1 = None
        convert_element_type_194: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_132: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_194, [4096, 1536])
        permute_66: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg103_1, view_132, permute_66);  arg103_1 = view_132 = permute_66 = None
        view_133: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [8, 512, 1536]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_134: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [8, 512, 24, -1]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_67: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        clone_43: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_135: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [-1, 512, 64]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_136: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_194, [4096, 1536])
        permute_68: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_37: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg105_1, view_136, permute_68);  arg105_1 = view_136 = permute_68 = None
        view_137: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [8, 512, 1536]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_138: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [8, 512, 24, -1]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_69: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_44: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_139: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [-1, 512, 64]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_72: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
        full_default_19: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_12: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_72, full_default_19);  permute_72 = full_default_19 = None
        bmm_12: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_135, div_12);  view_135 = div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_144: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [-1, 24, 512, 512]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_6: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_20, full_default_21, view_144);  full_default_20 = full_default_21 = view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_208: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        amax_6: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_208, [-1], True)
        sub_19: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_208, amax_6);  convert_element_type_208 = amax_6 = None
        exp_6: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_209: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_145: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_209, [-1, 512, 512]);  convert_element_type_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_140: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_194, [4096, 1536])
        permute_70: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_38: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg107_1, view_140, permute_70);  arg107_1 = view_140 = permute_70 = None
        view_141: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [8, 512, 1536]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_142: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_141, [8, 512, 24, -1]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_71: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        clone_45: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_143: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [-1, 512, 64]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_13: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_145, view_143);  view_145 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_146: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [-1, 24, 512, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_73: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_47: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_147: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [8, 512, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [4096, 1536]);  view_147 = None
        permute_74: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_39: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg109_1, view_148, permute_74);  arg109_1 = view_148 = permute_74 = None
        view_149: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [8, 512, 1536]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_45: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_149, convert_element_type_194);  view_149 = convert_element_type_194 = None
        convert_element_type_215: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_215, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_215, getitem_27);  convert_element_type_215 = getitem_27 = None
        add_46: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-07);  getitem_26 = None
        rsqrt_13: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_53: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_54: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
        add_47: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_54, arg111_1);  mul_54 = arg111_1 = None
        convert_element_type_216: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_216, [4096, 1536])
        permute_75: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_40: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg113_1, view_150, permute_75);  arg113_1 = view_150 = permute_75 = None
        view_151: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [8, 512, 6144]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_220: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_55: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_220, 0.5)
        mul_56: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_220, 0.7071067811865476);  convert_element_type_220 = None
        erf_6: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
        add_48: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_57: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, add_48);  mul_55 = add_48 = None
        convert_element_type_221: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_221, [4096, 6144]);  convert_element_type_221 = None
        permute_76: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_41: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg115_1, view_152, permute_76);  arg115_1 = view_152 = permute_76 = None
        view_153: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [8, 512, 1536]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_49: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_153, convert_element_type_216);  view_153 = convert_element_type_216 = None
        convert_element_type_225: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.float32);  add_49 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_225, [2], correction = 0, keepdim = True)
        getitem_28: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant7: "f32[][]cpu" = self._tensor_constant7
        lift_fresh_copy_7: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
        mul_60: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_7, 1);  lift_fresh_copy_7 = None
        sqrt_7: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_60);  mul_60 = sqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_239: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_23: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_21: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_225, getitem_29);  convert_element_type_225 = getitem_29 = None
        add_50: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-07);  getitem_28 = None
        rsqrt_14: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_58: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_59: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
        add_51: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_59, arg117_1);  mul_59 = arg117_1 = None
        convert_element_type_226: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_154: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_226, [4096, 1536])
        permute_77: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_42: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg119_1, view_154, permute_77);  arg119_1 = view_154 = permute_77 = None
        view_155: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [8, 512, 1536]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_156: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [8, 512, 24, -1]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_78: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        clone_50: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_157: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [-1, 512, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_158: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_226, [4096, 1536])
        permute_79: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_43: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg121_1, view_158, permute_79);  arg121_1 = view_158 = permute_79 = None
        view_159: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [8, 512, 1536]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_160: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_159, [8, 512, 24, -1]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_80: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_51: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_161: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [-1, 512, 64]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_83: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
        full_default_22: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_14: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_83, full_default_22);  permute_83 = full_default_22 = None
        bmm_14: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_157, div_14);  view_157 = div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_166: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [-1, 24, 512, 512]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_7: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_23, full_default_24, view_166);  full_default_23 = full_default_24 = view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_240: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        amax_7: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_240, [-1], True)
        sub_22: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, amax_7);  convert_element_type_240 = amax_7 = None
        exp_7: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_241: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_167: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [-1, 512, 512]);  convert_element_type_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_162: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_226, [4096, 1536])
        permute_81: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_44: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg123_1, view_162, permute_81);  arg123_1 = view_162 = permute_81 = None
        view_163: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [8, 512, 1536]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_164: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_163, [8, 512, 24, -1]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_82: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        clone_52: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_165: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [-1, 512, 64]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_15: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_167, view_165);  view_167 = view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_168: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [-1, 24, 512, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_84: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_54: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_169: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [8, 512, -1]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [4096, 1536]);  view_169 = None
        permute_85: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_45: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg125_1, view_170, permute_85);  arg125_1 = view_170 = permute_85 = None
        view_171: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [8, 512, 1536]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_52: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_171, convert_element_type_226);  view_171 = convert_element_type_226 = None
        convert_element_type_247: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.float32);  add_52 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_247, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_247, getitem_31);  convert_element_type_247 = getitem_31 = None
        add_53: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-07);  getitem_30 = None
        rsqrt_15: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_61: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_62: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
        add_54: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, arg127_1);  mul_62 = arg127_1 = None
        convert_element_type_248: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_248, [4096, 1536])
        permute_86: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_46: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg129_1, view_172, permute_86);  arg129_1 = view_172 = permute_86 = None
        view_173: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [8, 512, 6144]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_252: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_63: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_252, 0.5)
        mul_64: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_252, 0.7071067811865476);  convert_element_type_252 = None
        erf_7: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_55: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_65: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, add_55);  mul_63 = add_55 = None
        convert_element_type_253: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_253, [4096, 6144]);  convert_element_type_253 = None
        permute_87: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_47: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg131_1, view_174, permute_87);  arg131_1 = view_174 = permute_87 = None
        view_175: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [8, 512, 1536]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_56: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_175, convert_element_type_248);  view_175 = convert_element_type_248 = None
        convert_element_type_257: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.float32);  add_56 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_257, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant8: "f32[][]cpu" = self._tensor_constant8
        lift_fresh_copy_8: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
        mul_68: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_8, 1);  lift_fresh_copy_8 = None
        sqrt_8: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_68);  mul_68 = sqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_271: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_26: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_24: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_257, getitem_33);  convert_element_type_257 = getitem_33 = None
        add_57: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-07);  getitem_32 = None
        rsqrt_16: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_66: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_67: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
        add_58: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg133_1);  mul_67 = arg133_1 = None
        convert_element_type_258: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.bfloat16);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_176: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_258, [4096, 1536])
        permute_88: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_48: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg135_1, view_176, permute_88);  arg135_1 = view_176 = permute_88 = None
        view_177: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [8, 512, 1536]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_178: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [8, 512, 24, -1]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_89: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        clone_57: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_179: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [-1, 512, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_180: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_258, [4096, 1536])
        permute_90: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_49: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg137_1, view_180, permute_90);  arg137_1 = view_180 = permute_90 = None
        view_181: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [8, 512, 1536]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_182: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_181, [8, 512, 24, -1]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_91: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_58: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_183: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [-1, 512, 64]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_94: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        full_default_25: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_16: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_94, full_default_25);  permute_94 = full_default_25 = None
        bmm_16: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_179, div_16);  view_179 = div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_188: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [-1, 24, 512, 512]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_8: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_26, full_default_27, view_188);  full_default_26 = full_default_27 = view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_272: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        amax_8: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_272, [-1], True)
        sub_25: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, amax_8);  convert_element_type_272 = amax_8 = None
        exp_8: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_273: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_189: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_273, [-1, 512, 512]);  convert_element_type_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_184: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_258, [4096, 1536])
        permute_92: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_50: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg139_1, view_184, permute_92);  arg139_1 = view_184 = permute_92 = None
        view_185: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [8, 512, 1536]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_186: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [8, 512, 24, -1]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_93: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_59: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_187: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [-1, 512, 64]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_17: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_189, view_187);  view_189 = view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_190: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [-1, 24, 512, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_95: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_61: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_191: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [8, 512, -1]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [4096, 1536]);  view_191 = None
        permute_96: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_51: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg141_1, view_192, permute_96);  arg141_1 = view_192 = permute_96 = None
        view_193: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [8, 512, 1536]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, convert_element_type_258);  view_193 = convert_element_type_258 = None
        convert_element_type_279: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_279, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_279, getitem_35);  convert_element_type_279 = getitem_35 = None
        add_60: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-07);  getitem_34 = None
        rsqrt_17: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_69: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_70: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_69, arg142_1);  mul_69 = arg142_1 = None
        add_61: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_70, arg143_1);  mul_70 = arg143_1 = None
        convert_element_type_280: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_280, [4096, 1536])
        permute_97: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_52: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg145_1, view_194, permute_97);  arg145_1 = view_194 = permute_97 = None
        view_195: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [8, 512, 6144]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_284: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_71: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.5)
        mul_72: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.7071067811865476);  convert_element_type_284 = None
        erf_8: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
        add_62: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_73: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, add_62);  mul_71 = add_62 = None
        convert_element_type_285: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_285, [4096, 6144]);  convert_element_type_285 = None
        permute_98: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_53: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg147_1, view_196, permute_98);  arg147_1 = view_196 = permute_98 = None
        view_197: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [8, 512, 1536]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_197, convert_element_type_280);  view_197 = convert_element_type_280 = None
        convert_element_type_289: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_289, [2], correction = 0, keepdim = True)
        getitem_36: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant9: "f32[][]cpu" = self._tensor_constant9
        lift_fresh_copy_9: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
        mul_76: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_9, 1);  lift_fresh_copy_9 = None
        sqrt_9: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_76);  mul_76 = sqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_303: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_29: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_27: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_289, getitem_37);  convert_element_type_289 = getitem_37 = None
        add_64: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-07);  getitem_36 = None
        rsqrt_18: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_74: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_75: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, arg148_1);  mul_74 = arg148_1 = None
        add_65: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_75, arg149_1);  mul_75 = arg149_1 = None
        convert_element_type_290: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_198: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_290, [4096, 1536])
        permute_99: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_54: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg151_1, view_198, permute_99);  arg151_1 = view_198 = permute_99 = None
        view_199: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [8, 512, 1536]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_200: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [8, 512, 24, -1]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_100: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        clone_64: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_201: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [-1, 512, 64]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_202: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_290, [4096, 1536])
        permute_101: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_55: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg153_1, view_202, permute_101);  arg153_1 = view_202 = permute_101 = None
        view_203: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [8, 512, 1536]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_204: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_203, [8, 512, 24, -1]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_102: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        clone_65: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_205: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [-1, 512, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_105: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
        full_default_28: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_18: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_105, full_default_28);  permute_105 = full_default_28 = None
        bmm_18: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_201, div_18);  view_201 = div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_210: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [-1, 24, 512, 512]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_9: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_29, full_default_30, view_210);  full_default_29 = full_default_30 = view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_304: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        amax_9: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_304, [-1], True)
        sub_28: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_304, amax_9);  convert_element_type_304 = amax_9 = None
        exp_9: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_305: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_211: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_305, [-1, 512, 512]);  convert_element_type_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_206: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_290, [4096, 1536])
        permute_103: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_56: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg155_1, view_206, permute_103);  arg155_1 = view_206 = permute_103 = None
        view_207: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [8, 512, 1536]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_208: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_207, [8, 512, 24, -1]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_104: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        clone_66: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_209: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [-1, 512, 64]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_19: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_211, view_209);  view_211 = view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_212: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [-1, 24, 512, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_106: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_68: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_213: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [8, 512, -1]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [4096, 1536]);  view_213 = None
        permute_107: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_57: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg157_1, view_214, permute_107);  arg157_1 = view_214 = permute_107 = None
        view_215: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [8, 512, 1536]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_66: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_215, convert_element_type_290);  view_215 = convert_element_type_290 = None
        convert_element_type_311: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.float32);  add_66 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_311, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_311, getitem_39);  convert_element_type_311 = getitem_39 = None
        add_67: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-07);  getitem_38 = None
        rsqrt_19: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_77: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_78: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, arg158_1);  mul_77 = arg158_1 = None
        add_68: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, arg159_1);  mul_78 = arg159_1 = None
        convert_element_type_312: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_312, [4096, 1536])
        permute_108: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_58: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg161_1, view_216, permute_108);  arg161_1 = view_216 = permute_108 = None
        view_217: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [8, 512, 6144]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_316: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_79: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_316, 0.5)
        mul_80: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_316, 0.7071067811865476);  convert_element_type_316 = None
        erf_9: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_80);  mul_80 = None
        add_69: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_81: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, add_69);  mul_79 = add_69 = None
        convert_element_type_317: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_317, [4096, 6144]);  convert_element_type_317 = None
        permute_109: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_59: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg163_1, view_218, permute_109);  arg163_1 = view_218 = permute_109 = None
        view_219: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [8, 512, 1536]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_70: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_219, convert_element_type_312);  view_219 = convert_element_type_312 = None
        convert_element_type_321: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.float32);  add_70 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_321, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant10: "f32[][]cpu" = self._tensor_constant10
        lift_fresh_copy_10: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
        mul_84: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_10, 1);  lift_fresh_copy_10 = None
        sqrt_10: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_84);  mul_84 = sqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_335: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_32: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_30: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_321, getitem_41);  convert_element_type_321 = getitem_41 = None
        add_71: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-07);  getitem_40 = None
        rsqrt_20: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_82: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_83: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, arg164_1);  mul_82 = arg164_1 = None
        add_72: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, arg165_1);  mul_83 = arg165_1 = None
        convert_element_type_322: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_220: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_322, [4096, 1536])
        permute_110: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_60: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg167_1, view_220, permute_110);  arg167_1 = view_220 = permute_110 = None
        view_221: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [8, 512, 1536]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_222: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [8, 512, 24, -1]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_111: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        clone_71: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_223: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [-1, 512, 64]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_224: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_322, [4096, 1536])
        permute_112: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_61: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg169_1, view_224, permute_112);  arg169_1 = view_224 = permute_112 = None
        view_225: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [8, 512, 1536]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_226: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [8, 512, 24, -1]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_113: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_72: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_227: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [-1, 512, 64]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_116: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
        full_default_31: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_20: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_116, full_default_31);  permute_116 = full_default_31 = None
        bmm_20: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_223, div_20);  view_223 = div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_232: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [-1, 24, 512, 512]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_10: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_32, full_default_33, view_232);  full_default_32 = full_default_33 = view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_336: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        amax_10: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_336, [-1], True)
        sub_31: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_336, amax_10);  convert_element_type_336 = amax_10 = None
        exp_10: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_337: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_233: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [-1, 512, 512]);  convert_element_type_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_228: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_322, [4096, 1536])
        permute_114: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_62: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg171_1, view_228, permute_114);  arg171_1 = view_228 = permute_114 = None
        view_229: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [8, 512, 1536]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_230: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_229, [8, 512, 24, -1]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_115: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
        clone_73: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_231: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [-1, 512, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_21: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_233, view_231);  view_233 = view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_234: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [-1, 24, 512, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_117: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_75: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_235: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [8, 512, -1]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [4096, 1536]);  view_235 = None
        permute_118: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_63: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg173_1, view_236, permute_118);  arg173_1 = view_236 = permute_118 = None
        view_237: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [8, 512, 1536]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_73: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_237, convert_element_type_322);  view_237 = convert_element_type_322 = None
        convert_element_type_343: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.float32);  add_73 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_343, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_343, getitem_43);  convert_element_type_343 = getitem_43 = None
        add_74: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-07);  getitem_42 = None
        rsqrt_21: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_85: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_86: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, arg174_1);  mul_85 = arg174_1 = None
        add_75: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, arg175_1);  mul_86 = arg175_1 = None
        convert_element_type_344: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_344, [4096, 1536])
        permute_119: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_64: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg177_1, view_238, permute_119);  arg177_1 = view_238 = permute_119 = None
        view_239: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [8, 512, 6144]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_348: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_87: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_348, 0.5)
        mul_88: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_348, 0.7071067811865476);  convert_element_type_348 = None
        erf_10: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_76: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_89: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, add_76);  mul_87 = add_76 = None
        convert_element_type_349: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_349, [4096, 6144]);  convert_element_type_349 = None
        permute_120: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_65: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg179_1, view_240, permute_120);  arg179_1 = view_240 = permute_120 = None
        view_241: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [8, 512, 1536]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_77: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_241, convert_element_type_344);  view_241 = convert_element_type_344 = None
        convert_element_type_353: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.float32);  add_77 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_353, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant11: "f32[][]cpu" = self._tensor_constant11
        lift_fresh_copy_11: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
        mul_92: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_11, 1);  lift_fresh_copy_11 = None
        sqrt_11: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_92);  mul_92 = sqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_367: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_35: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_33: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_353, getitem_45);  convert_element_type_353 = getitem_45 = None
        add_78: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-07);  getitem_44 = None
        rsqrt_22: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_90: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_91: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, arg180_1);  mul_90 = arg180_1 = None
        add_79: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, arg181_1);  mul_91 = arg181_1 = None
        convert_element_type_354: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_242: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_354, [4096, 1536])
        permute_121: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_66: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg183_1, view_242, permute_121);  arg183_1 = view_242 = permute_121 = None
        view_243: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [8, 512, 1536]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_244: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_243, [8, 512, 24, -1]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_122: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        clone_78: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_245: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [-1, 512, 64]);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_246: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_354, [4096, 1536])
        permute_123: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_67: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg185_1, view_246, permute_123);  arg185_1 = view_246 = permute_123 = None
        view_247: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [8, 512, 1536]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_248: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_247, [8, 512, 24, -1]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_124: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        clone_79: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_249: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [-1, 512, 64]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_127: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
        full_default_34: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_22: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_127, full_default_34);  permute_127 = full_default_34 = None
        bmm_22: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_245, div_22);  view_245 = div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_254: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [-1, 24, 512, 512]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_11: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_35, full_default_36, view_254);  full_default_35 = full_default_36 = view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_368: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        amax_11: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_368, [-1], True)
        sub_34: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_368, amax_11);  convert_element_type_368 = amax_11 = None
        exp_11: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_369: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_255: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_369, [-1, 512, 512]);  convert_element_type_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_250: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_354, [4096, 1536])
        permute_125: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_68: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg187_1, view_250, permute_125);  arg187_1 = view_250 = permute_125 = None
        view_251: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [8, 512, 1536]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_252: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [8, 512, 24, -1]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_126: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        clone_80: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_253: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [-1, 512, 64]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_23: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_255, view_253);  view_255 = view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_256: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [-1, 24, 512, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_128: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_82: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_257: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [8, 512, -1]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [4096, 1536]);  view_257 = None
        permute_129: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_69: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg189_1, view_258, permute_129);  arg189_1 = view_258 = permute_129 = None
        view_259: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [8, 512, 1536]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_80: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_259, convert_element_type_354);  view_259 = convert_element_type_354 = None
        convert_element_type_375: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.float32);  add_80 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_375, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_375, getitem_47);  convert_element_type_375 = getitem_47 = None
        add_81: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-07);  getitem_46 = None
        rsqrt_23: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        mul_93: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_94: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, arg190_1);  mul_93 = arg190_1 = None
        add_82: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_94, arg191_1);  mul_94 = arg191_1 = None
        convert_element_type_376: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_376, [4096, 1536])
        permute_130: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_70: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg193_1, view_260, permute_130);  arg193_1 = view_260 = permute_130 = None
        view_261: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [8, 512, 6144]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_380: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_95: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, 0.5)
        mul_96: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, 0.7071067811865476);  convert_element_type_380 = None
        erf_11: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
        add_83: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_97: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, add_83);  mul_95 = add_83 = None
        convert_element_type_381: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_97, torch.bfloat16);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_381, [4096, 6144]);  convert_element_type_381 = None
        permute_131: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_71: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg195_1, view_262, permute_131);  arg195_1 = view_262 = permute_131 = None
        view_263: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [8, 512, 1536]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_84: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_263, convert_element_type_376);  view_263 = convert_element_type_376 = None
        convert_element_type_385: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.float32);  add_84 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_385, [2], correction = 0, keepdim = True)
        getitem_48: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant12: "f32[][]cpu" = self._tensor_constant12
        lift_fresh_copy_12: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
        mul_100: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_12, 1);  lift_fresh_copy_12 = None
        sqrt_12: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_100);  mul_100 = sqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_399: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_38: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_36: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_385, getitem_49);  convert_element_type_385 = getitem_49 = None
        add_85: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-07);  getitem_48 = None
        rsqrt_24: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_98: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_99: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, arg196_1);  mul_98 = arg196_1 = None
        add_86: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, arg197_1);  mul_99 = arg197_1 = None
        convert_element_type_386: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_264: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_386, [4096, 1536])
        permute_132: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_72: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg199_1, view_264, permute_132);  arg199_1 = view_264 = permute_132 = None
        view_265: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [8, 512, 1536]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_266: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_265, [8, 512, 24, -1]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_133: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        clone_85: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
        view_267: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [-1, 512, 64]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_268: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_386, [4096, 1536])
        permute_134: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_73: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg201_1, view_268, permute_134);  arg201_1 = view_268 = permute_134 = None
        view_269: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [8, 512, 1536]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_270: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_269, [8, 512, 24, -1]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_135: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        clone_86: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_271: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [-1, 512, 64]);  clone_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_138: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
        full_default_37: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_24: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_138, full_default_37);  permute_138 = full_default_37 = None
        bmm_24: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_267, div_24);  view_267 = div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_276: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [-1, 24, 512, 512]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_12: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_38, full_default_39, view_276);  full_default_38 = full_default_39 = view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_400: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32);  where_12 = None
        amax_12: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_400, [-1], True)
        sub_37: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_400, amax_12);  convert_element_type_400 = amax_12 = None
        exp_12: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_25: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_401: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_277: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_401, [-1, 512, 512]);  convert_element_type_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_272: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_386, [4096, 1536])
        permute_136: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_74: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg203_1, view_272, permute_136);  arg203_1 = view_272 = permute_136 = None
        view_273: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [8, 512, 1536]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_274: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_273, [8, 512, 24, -1]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_137: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_87: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_275: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [-1, 512, 64]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_25: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_277, view_275);  view_277 = view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_278: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [-1, 24, 512, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_139: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_89: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_279: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [8, 512, -1]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_279, [4096, 1536]);  view_279 = None
        permute_140: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_75: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg205_1, view_280, permute_140);  arg205_1 = view_280 = permute_140 = None
        view_281: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [8, 512, 1536]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_281, convert_element_type_386);  view_281 = convert_element_type_386 = None
        convert_element_type_407: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_407, [2], correction = 0, keepdim = True)
        getitem_50: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        sub_38: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_407, getitem_51);  convert_element_type_407 = getitem_51 = None
        add_88: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-07);  getitem_50 = None
        rsqrt_25: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_101: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
        mul_102: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, arg206_1);  mul_101 = arg206_1 = None
        add_89: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, arg207_1);  mul_102 = arg207_1 = None
        convert_element_type_408: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_282: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_408, [4096, 1536])
        permute_141: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_76: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg209_1, view_282, permute_141);  arg209_1 = view_282 = permute_141 = None
        view_283: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [8, 512, 6144]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_412: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        mul_103: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_412, 0.5)
        mul_104: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_412, 0.7071067811865476);  convert_element_type_412 = None
        erf_12: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_90: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_105: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, add_90);  mul_103 = add_90 = None
        convert_element_type_413: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_284: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_413, [4096, 6144]);  convert_element_type_413 = None
        permute_142: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_77: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg211_1, view_284, permute_142);  arg211_1 = view_284 = permute_142 = None
        view_285: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [8, 512, 1536]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_285, convert_element_type_408);  view_285 = convert_element_type_408 = None
        convert_element_type_417: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_417, [2], correction = 0, keepdim = True)
        getitem_52: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant13: "f32[][]cpu" = self._tensor_constant13
        lift_fresh_copy_13: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
        mul_108: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_13, 1);  lift_fresh_copy_13 = None
        sqrt_13: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_108);  mul_108 = sqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_431: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_41: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_39: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_417, getitem_53);  convert_element_type_417 = getitem_53 = None
        add_92: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-07);  getitem_52 = None
        rsqrt_26: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_106: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
        mul_107: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, arg212_1);  mul_106 = arg212_1 = None
        add_93: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_107, arg213_1);  mul_107 = arg213_1 = None
        convert_element_type_418: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_286: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_418, [4096, 1536])
        permute_143: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        addmm_78: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg215_1, view_286, permute_143);  arg215_1 = view_286 = permute_143 = None
        view_287: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [8, 512, 1536]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_288: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_287, [8, 512, 24, -1]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_144: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        clone_92: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_289: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [-1, 512, 64]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_290: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_418, [4096, 1536])
        permute_145: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_79: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg217_1, view_290, permute_145);  arg217_1 = view_290 = permute_145 = None
        view_291: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [8, 512, 1536]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_292: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_291, [8, 512, 24, -1]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_146: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_93: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_293: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [-1, 512, 64]);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_149: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
        full_default_40: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_26: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_149, full_default_40);  permute_149 = full_default_40 = None
        bmm_26: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_289, div_26);  view_289 = div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_298: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [-1, 24, 512, 512]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_13: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_41, full_default_42, view_298);  full_default_41 = full_default_42 = view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_432: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        amax_13: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_432, [-1], True)
        sub_40: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_432, amax_13);  convert_element_type_432 = amax_13 = None
        exp_13: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_27: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_433: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_299: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_433, [-1, 512, 512]);  convert_element_type_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_294: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_418, [4096, 1536])
        permute_147: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_80: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg219_1, view_294, permute_147);  arg219_1 = view_294 = permute_147 = None
        view_295: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [8, 512, 1536]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_296: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [8, 512, 24, -1]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_148: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        clone_94: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_297: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [-1, 512, 64]);  clone_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_27: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_299, view_297);  view_299 = view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_300: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [-1, 24, 512, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_150: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_96: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_301: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [8, 512, -1]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_302: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [4096, 1536]);  view_301 = None
        permute_151: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_81: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg221_1, view_302, permute_151);  arg221_1 = view_302 = permute_151 = None
        view_303: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [8, 512, 1536]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_94: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_303, convert_element_type_418);  view_303 = convert_element_type_418 = None
        convert_element_type_439: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.float32);  add_94 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_439, [2], correction = 0, keepdim = True)
        getitem_54: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        sub_41: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_439, getitem_55);  convert_element_type_439 = getitem_55 = None
        add_95: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-07);  getitem_54 = None
        rsqrt_27: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_109: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_110: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, arg222_1);  mul_109 = arg222_1 = None
        add_96: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_110, arg223_1);  mul_110 = arg223_1 = None
        convert_element_type_440: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_304: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_440, [4096, 1536])
        permute_152: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_82: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg225_1, view_304, permute_152);  arg225_1 = view_304 = permute_152 = None
        view_305: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [8, 512, 6144]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_444: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        mul_111: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, 0.5)
        mul_112: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, 0.7071067811865476);  convert_element_type_444 = None
        erf_13: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_97: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_113: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, add_97);  mul_111 = add_97 = None
        convert_element_type_445: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_306: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_445, [4096, 6144]);  convert_element_type_445 = None
        permute_153: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_83: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg227_1, view_306, permute_153);  arg227_1 = view_306 = permute_153 = None
        view_307: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [8, 512, 1536]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_98: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_307, convert_element_type_440);  view_307 = convert_element_type_440 = None
        convert_element_type_449: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_98, torch.float32);  add_98 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_449, [2], correction = 0, keepdim = True)
        getitem_56: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant14: "f32[][]cpu" = self._tensor_constant14
        lift_fresh_copy_14: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
        mul_116: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_14, 1);  lift_fresh_copy_14 = None
        sqrt_14: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_116);  mul_116 = sqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_463: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_44: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_42: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_449, getitem_57);  convert_element_type_449 = getitem_57 = None
        add_99: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-07);  getitem_56 = None
        rsqrt_28: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_114: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
        mul_115: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, arg228_1);  mul_114 = arg228_1 = None
        add_100: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, arg229_1);  mul_115 = arg229_1 = None
        convert_element_type_450: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_308: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_450, [4096, 1536])
        permute_154: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        addmm_84: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg231_1, view_308, permute_154);  arg231_1 = view_308 = permute_154 = None
        view_309: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [8, 512, 1536]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_310: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [8, 512, 24, -1]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_155: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        clone_99: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_311: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [-1, 512, 64]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_312: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_450, [4096, 1536])
        permute_156: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_85: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg233_1, view_312, permute_156);  arg233_1 = view_312 = permute_156 = None
        view_313: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_85, [8, 512, 1536]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_314: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_313, [8, 512, 24, -1]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_157: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
        clone_100: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_315: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [-1, 512, 64]);  clone_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_160: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_315, [0, 2, 1]);  view_315 = None
        full_default_43: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_28: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_160, full_default_43);  permute_160 = full_default_43 = None
        bmm_28: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_311, div_28);  view_311 = div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_320: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [-1, 24, 512, 512]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_14: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_44, full_default_45, view_320);  full_default_44 = full_default_45 = view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_464: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        amax_14: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_464, [-1], True)
        sub_43: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_464, amax_14);  convert_element_type_464 = amax_14 = None
        exp_14: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_29: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_465: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_321: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_465, [-1, 512, 512]);  convert_element_type_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_316: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_450, [4096, 1536])
        permute_158: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_86: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg235_1, view_316, permute_158);  arg235_1 = view_316 = permute_158 = None
        view_317: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [8, 512, 1536]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_318: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_317, [8, 512, 24, -1]);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_159: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_101: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_319: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [-1, 512, 64]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_29: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_321, view_319);  view_321 = view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_322: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [-1, 24, 512, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_161: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_103: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_323: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [8, 512, -1]);  clone_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_324: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_323, [4096, 1536]);  view_323 = None
        permute_162: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_87: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg237_1, view_324, permute_162);  arg237_1 = view_324 = permute_162 = None
        view_325: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_87, [8, 512, 1536]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_101: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_325, convert_element_type_450);  view_325 = convert_element_type_450 = None
        convert_element_type_471: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.float32);  add_101 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_471, [2], correction = 0, keepdim = True)
        getitem_58: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        sub_44: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_471, getitem_59);  convert_element_type_471 = getitem_59 = None
        add_102: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-07);  getitem_58 = None
        rsqrt_29: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_117: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_118: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, arg238_1);  mul_117 = arg238_1 = None
        add_103: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_118, arg239_1);  mul_118 = arg239_1 = None
        convert_element_type_472: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_326: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_472, [4096, 1536])
        permute_163: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        addmm_88: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg241_1, view_326, permute_163);  arg241_1 = view_326 = permute_163 = None
        view_327: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [8, 512, 6144]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_476: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        mul_119: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_476, 0.5)
        mul_120: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_476, 0.7071067811865476);  convert_element_type_476 = None
        erf_14: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
        add_104: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_121: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, add_104);  mul_119 = add_104 = None
        convert_element_type_477: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_328: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_477, [4096, 6144]);  convert_element_type_477 = None
        permute_164: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_89: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg243_1, view_328, permute_164);  arg243_1 = view_328 = permute_164 = None
        view_329: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_89, [8, 512, 1536]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_105: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_329, convert_element_type_472);  view_329 = convert_element_type_472 = None
        convert_element_type_481: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_481, [2], correction = 0, keepdim = True)
        getitem_60: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant15: "f32[][]cpu" = self._tensor_constant15
        lift_fresh_copy_15: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
        mul_124: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_15, 1);  lift_fresh_copy_15 = None
        sqrt_15: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_124);  mul_124 = sqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_495: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_47: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_45: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_481, getitem_61);  convert_element_type_481 = getitem_61 = None
        add_106: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-07);  getitem_60 = None
        rsqrt_30: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_122: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
        mul_123: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, arg244_1);  mul_122 = arg244_1 = None
        add_107: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, arg245_1);  mul_123 = arg245_1 = None
        convert_element_type_482: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_330: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_482, [4096, 1536])
        permute_165: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        addmm_90: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg247_1, view_330, permute_165);  arg247_1 = view_330 = permute_165 = None
        view_331: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [8, 512, 1536]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_332: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_331, [8, 512, 24, -1]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_166: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        clone_106: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_333: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [-1, 512, 64]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_334: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_482, [4096, 1536])
        permute_167: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_91: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg249_1, view_334, permute_167);  arg249_1 = view_334 = permute_167 = None
        view_335: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_91, [8, 512, 1536]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_336: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_335, [8, 512, 24, -1]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_168: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
        clone_107: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_337: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [-1, 512, 64]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_171: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
        full_default_46: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_30: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_171, full_default_46);  permute_171 = full_default_46 = None
        bmm_30: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_333, div_30);  view_333 = div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_342: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [-1, 24, 512, 512]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_15: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_47, full_default_48, view_342);  full_default_47 = full_default_48 = view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_496: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        amax_15: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_496, [-1], True)
        sub_46: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_496, amax_15);  convert_element_type_496 = amax_15 = None
        exp_15: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_31: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_497: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_343: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_497, [-1, 512, 512]);  convert_element_type_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_338: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_482, [4096, 1536])
        permute_169: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_92: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg251_1, view_338, permute_169);  arg251_1 = view_338 = permute_169 = None
        view_339: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_92, [8, 512, 1536]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_340: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_339, [8, 512, 24, -1]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_170: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        clone_108: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_341: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [-1, 512, 64]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_31: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_343, view_341);  view_343 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_344: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [-1, 24, 512, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_172: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_110: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_345: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [8, 512, -1]);  clone_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_346: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [4096, 1536]);  view_345 = None
        permute_173: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        addmm_93: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg253_1, view_346, permute_173);  arg253_1 = view_346 = permute_173 = None
        view_347: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_93, [8, 512, 1536]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_108: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_347, convert_element_type_482);  view_347 = convert_element_type_482 = None
        convert_element_type_503: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.float32);  add_108 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_503, [2], correction = 0, keepdim = True)
        getitem_62: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        sub_47: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_503, getitem_63);  convert_element_type_503 = getitem_63 = None
        add_109: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-07);  getitem_62 = None
        rsqrt_31: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_125: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_126: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, arg254_1);  mul_125 = arg254_1 = None
        add_110: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_126, arg255_1);  mul_126 = arg255_1 = None
        convert_element_type_504: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_110, torch.bfloat16);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_348: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_504, [4096, 1536])
        permute_174: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        addmm_94: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg257_1, view_348, permute_174);  arg257_1 = view_348 = permute_174 = None
        view_349: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [8, 512, 6144]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_508: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        mul_127: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_508, 0.5)
        mul_128: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_508, 0.7071067811865476);  convert_element_type_508 = None
        erf_15: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
        add_111: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_129: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, add_111);  mul_127 = add_111 = None
        convert_element_type_509: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_350: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_509, [4096, 6144]);  convert_element_type_509 = None
        permute_175: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        addmm_95: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg259_1, view_350, permute_175);  arg259_1 = view_350 = permute_175 = None
        view_351: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_95, [8, 512, 1536]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_112: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_351, convert_element_type_504);  view_351 = convert_element_type_504 = None
        convert_element_type_513: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.float32);  add_112 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_513, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant16: "f32[][]cpu" = self._tensor_constant16
        lift_fresh_copy_16: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
        mul_132: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_16, 1);  lift_fresh_copy_16 = None
        sqrt_16: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_132);  mul_132 = sqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_527: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_50: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_48: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_513, getitem_65);  convert_element_type_513 = getitem_65 = None
        add_113: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-07);  getitem_64 = None
        rsqrt_32: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_130: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
        mul_131: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, arg260_1);  mul_130 = arg260_1 = None
        add_114: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, arg261_1);  mul_131 = arg261_1 = None
        convert_element_type_514: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_352: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_514, [4096, 1536])
        permute_176: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_96: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg263_1, view_352, permute_176);  arg263_1 = view_352 = permute_176 = None
        view_353: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_96, [8, 512, 1536]);  addmm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_354: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_353, [8, 512, 24, -1]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_177: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        clone_113: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_355: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [-1, 512, 64]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_356: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_514, [4096, 1536])
        permute_178: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_97: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg265_1, view_356, permute_178);  arg265_1 = view_356 = permute_178 = None
        view_357: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_97, [8, 512, 1536]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_358: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_357, [8, 512, 24, -1]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_179: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        clone_114: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_359: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [-1, 512, 64]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_182: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None
        full_default_49: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_32: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_182, full_default_49);  permute_182 = full_default_49 = None
        bmm_32: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_355, div_32);  view_355 = div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_364: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [-1, 24, 512, 512]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_16: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_50, full_default_51, view_364);  full_default_50 = full_default_51 = view_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_528: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        amax_16: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_528, [-1], True)
        sub_49: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_528, amax_16);  convert_element_type_528 = amax_16 = None
        exp_16: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_33: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_529: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16);  div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_365: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_529, [-1, 512, 512]);  convert_element_type_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_360: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_514, [4096, 1536])
        permute_180: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_98: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg267_1, view_360, permute_180);  arg267_1 = view_360 = permute_180 = None
        view_361: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_98, [8, 512, 1536]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_362: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_361, [8, 512, 24, -1]);  view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_181: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
        clone_115: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_363: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [-1, 512, 64]);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_33: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_365, view_363);  view_365 = view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_366: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [-1, 24, 512, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_183: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_117: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_367: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [8, 512, -1]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_368: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_367, [4096, 1536]);  view_367 = None
        permute_184: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_99: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg269_1, view_368, permute_184);  arg269_1 = view_368 = permute_184 = None
        view_369: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_99, [8, 512, 1536]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_115: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_369, convert_element_type_514);  view_369 = convert_element_type_514 = None
        convert_element_type_535: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.float32);  add_115 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_535, [2], correction = 0, keepdim = True)
        getitem_66: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        sub_50: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_535, getitem_67);  convert_element_type_535 = getitem_67 = None
        add_116: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-07);  getitem_66 = None
        rsqrt_33: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_133: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
        mul_134: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, arg270_1);  mul_133 = arg270_1 = None
        add_117: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, arg271_1);  mul_134 = arg271_1 = None
        convert_element_type_536: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_370: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_536, [4096, 1536])
        permute_185: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        addmm_100: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg273_1, view_370, permute_185);  arg273_1 = view_370 = permute_185 = None
        view_371: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_100, [8, 512, 6144]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_540: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        mul_135: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_540, 0.5)
        mul_136: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_540, 0.7071067811865476);  convert_element_type_540 = None
        erf_16: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_136);  mul_136 = None
        add_118: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_137: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, add_118);  mul_135 = add_118 = None
        convert_element_type_541: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_372: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_541, [4096, 6144]);  convert_element_type_541 = None
        permute_186: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_101: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg275_1, view_372, permute_186);  arg275_1 = view_372 = permute_186 = None
        view_373: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_101, [8, 512, 1536]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_119: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_373, convert_element_type_536);  view_373 = convert_element_type_536 = None
        convert_element_type_545: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.float32);  add_119 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_545, [2], correction = 0, keepdim = True)
        getitem_68: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant17: "f32[][]cpu" = self._tensor_constant17
        lift_fresh_copy_17: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
        mul_140: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_17, 1);  lift_fresh_copy_17 = None
        sqrt_17: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_140);  mul_140 = sqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_559: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_53: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_51: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_545, getitem_69);  convert_element_type_545 = getitem_69 = None
        add_120: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-07);  getitem_68 = None
        rsqrt_34: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_138: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
        mul_139: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, arg276_1);  mul_138 = arg276_1 = None
        add_121: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, arg277_1);  mul_139 = arg277_1 = None
        convert_element_type_546: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_374: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_546, [4096, 1536])
        permute_187: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_102: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg279_1, view_374, permute_187);  arg279_1 = view_374 = permute_187 = None
        view_375: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_102, [8, 512, 1536]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_376: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_375, [8, 512, 24, -1]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_188: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        clone_120: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_377: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [-1, 512, 64]);  clone_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_378: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_546, [4096, 1536])
        permute_189: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_103: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg281_1, view_378, permute_189);  arg281_1 = view_378 = permute_189 = None
        view_379: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_103, [8, 512, 1536]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_380: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_379, [8, 512, 24, -1]);  view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_190: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
        clone_121: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_381: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [-1, 512, 64]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_193: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
        full_default_52: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_34: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_193, full_default_52);  permute_193 = full_default_52 = None
        bmm_34: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_377, div_34);  view_377 = div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_386: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [-1, 24, 512, 512]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_17: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_53, full_default_54, view_386);  full_default_53 = full_default_54 = view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_560: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        amax_17: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_560, [-1], True)
        sub_52: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_560, amax_17);  convert_element_type_560 = amax_17 = None
        exp_17: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_35: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_561: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_35, torch.bfloat16);  div_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_387: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_561, [-1, 512, 512]);  convert_element_type_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_382: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_546, [4096, 1536])
        permute_191: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_104: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg283_1, view_382, permute_191);  arg283_1 = view_382 = permute_191 = None
        view_383: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_104, [8, 512, 1536]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_384: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_383, [8, 512, 24, -1]);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_192: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        clone_122: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_385: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [-1, 512, 64]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_35: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_387, view_385);  view_387 = view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_388: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [-1, 24, 512, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_194: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_124: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_389: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [8, 512, -1]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_390: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_389, [4096, 1536]);  view_389 = None
        permute_195: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_105: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg285_1, view_390, permute_195);  arg285_1 = view_390 = permute_195 = None
        view_391: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_105, [8, 512, 1536]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_122: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_391, convert_element_type_546);  view_391 = convert_element_type_546 = None
        convert_element_type_567: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.float32);  add_122 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_567, [2], correction = 0, keepdim = True)
        getitem_70: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        sub_53: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_567, getitem_71);  convert_element_type_567 = getitem_71 = None
        add_123: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-07);  getitem_70 = None
        rsqrt_35: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_141: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
        mul_142: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, arg286_1);  mul_141 = arg286_1 = None
        add_124: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, arg287_1);  mul_142 = arg287_1 = None
        convert_element_type_568: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_568, [4096, 1536])
        permute_196: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        addmm_106: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg289_1, view_392, permute_196);  arg289_1 = view_392 = permute_196 = None
        view_393: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_106, [8, 512, 6144]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_572: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_143: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_572, 0.5)
        mul_144: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_572, 0.7071067811865476);  convert_element_type_572 = None
        erf_17: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
        add_125: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_145: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, add_125);  mul_143 = add_125 = None
        convert_element_type_573: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_394: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_573, [4096, 6144]);  convert_element_type_573 = None
        permute_197: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        addmm_107: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg291_1, view_394, permute_197);  arg291_1 = view_394 = permute_197 = None
        view_395: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_107, [8, 512, 1536]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_126: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_395, convert_element_type_568);  view_395 = convert_element_type_568 = None
        convert_element_type_577: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_126, torch.float32);  add_126 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_577, [2], correction = 0, keepdim = True)
        getitem_72: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant18: "f32[][]cpu" = self._tensor_constant18
        lift_fresh_copy_18: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
        mul_148: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_18, 1);  lift_fresh_copy_18 = None
        sqrt_18: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_148);  mul_148 = sqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_591: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_56: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_54: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_577, getitem_73);  convert_element_type_577 = getitem_73 = None
        add_127: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-07);  getitem_72 = None
        rsqrt_36: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        mul_146: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
        mul_147: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, arg292_1);  mul_146 = arg292_1 = None
        add_128: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_147, arg293_1);  mul_147 = arg293_1 = None
        convert_element_type_578: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_396: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_578, [4096, 1536])
        permute_198: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        addmm_108: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg295_1, view_396, permute_198);  arg295_1 = view_396 = permute_198 = None
        view_397: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_108, [8, 512, 1536]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_398: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [8, 512, 24, -1]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_199: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        clone_127: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_399: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [-1, 512, 64]);  clone_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_400: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_578, [4096, 1536])
        permute_200: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_109: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg297_1, view_400, permute_200);  arg297_1 = view_400 = permute_200 = None
        view_401: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_109, [8, 512, 1536]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_402: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_401, [8, 512, 24, -1]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_201: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_128: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_403: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [-1, 512, 64]);  clone_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_204: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
        full_default_55: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_36: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_204, full_default_55);  permute_204 = full_default_55 = None
        bmm_36: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_399, div_36);  view_399 = div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_408: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [-1, 24, 512, 512]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_18: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_56, full_default_57, view_408);  full_default_56 = full_default_57 = view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_592: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32);  where_18 = None
        amax_18: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_592, [-1], True)
        sub_55: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_592, amax_18);  convert_element_type_592 = amax_18 = None
        exp_18: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_37: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_593: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16);  div_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_409: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_593, [-1, 512, 512]);  convert_element_type_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_404: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_578, [4096, 1536])
        permute_202: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_110: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg299_1, view_404, permute_202);  arg299_1 = view_404 = permute_202 = None
        view_405: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_110, [8, 512, 1536]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_406: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_405, [8, 512, 24, -1]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_203: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        clone_129: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_407: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [-1, 512, 64]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_37: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_409, view_407);  view_409 = view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_410: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [-1, 24, 512, 64]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_205: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_131: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_411: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [8, 512, -1]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_412: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [4096, 1536]);  view_411 = None
        permute_206: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_111: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg301_1, view_412, permute_206);  arg301_1 = view_412 = permute_206 = None
        view_413: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_111, [8, 512, 1536]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_129: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_413, convert_element_type_578);  view_413 = convert_element_type_578 = None
        convert_element_type_599: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.float32);  add_129 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_599, [2], correction = 0, keepdim = True)
        getitem_74: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        sub_56: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_599, getitem_75);  convert_element_type_599 = getitem_75 = None
        add_130: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-07);  getitem_74 = None
        rsqrt_37: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_149: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
        mul_150: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_149, arg302_1);  mul_149 = arg302_1 = None
        add_131: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_150, arg303_1);  mul_150 = arg303_1 = None
        convert_element_type_600: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_414: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_600, [4096, 1536])
        permute_207: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
        addmm_112: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg305_1, view_414, permute_207);  arg305_1 = view_414 = permute_207 = None
        view_415: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_112, [8, 512, 6144]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_604: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_151: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_604, 0.5)
        mul_152: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_604, 0.7071067811865476);  convert_element_type_604 = None
        erf_18: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
        add_132: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_153: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, add_132);  mul_151 = add_132 = None
        convert_element_type_605: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_153, torch.bfloat16);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_416: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_605, [4096, 6144]);  convert_element_type_605 = None
        permute_208: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        addmm_113: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg307_1, view_416, permute_208);  arg307_1 = view_416 = permute_208 = None
        view_417: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_113, [8, 512, 1536]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_133: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_417, convert_element_type_600);  view_417 = convert_element_type_600 = None
        convert_element_type_609: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.float32);  add_133 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_609, [2], correction = 0, keepdim = True)
        getitem_76: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant19: "f32[][]cpu" = self._tensor_constant19
        lift_fresh_copy_19: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant19);  _tensor_constant19 = None
        mul_156: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_19, 1);  lift_fresh_copy_19 = None
        sqrt_19: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_156);  mul_156 = sqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_623: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_59: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_57: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_609, getitem_77);  convert_element_type_609 = getitem_77 = None
        add_134: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-07);  getitem_76 = None
        rsqrt_38: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_154: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
        mul_155: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, arg308_1);  mul_154 = arg308_1 = None
        add_135: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, arg309_1);  mul_155 = arg309_1 = None
        convert_element_type_610: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_418: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_610, [4096, 1536])
        permute_209: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        addmm_114: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg311_1, view_418, permute_209);  arg311_1 = view_418 = permute_209 = None
        view_419: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_114, [8, 512, 1536]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_420: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_419, [8, 512, 24, -1]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_210: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        clone_134: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_421: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_134, [-1, 512, 64]);  clone_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_422: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_610, [4096, 1536])
        permute_211: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        addmm_115: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg313_1, view_422, permute_211);  arg313_1 = view_422 = permute_211 = None
        view_423: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_115, [8, 512, 1536]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_424: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_423, [8, 512, 24, -1]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_212: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        clone_135: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_425: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [-1, 512, 64]);  clone_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_215: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        full_default_58: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_38: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_215, full_default_58);  permute_215 = full_default_58 = None
        bmm_38: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_421, div_38);  view_421 = div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_430: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [-1, 24, 512, 512]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_19: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_59, full_default_60, view_430);  full_default_59 = full_default_60 = view_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_624: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        amax_19: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_624, [-1], True)
        sub_58: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_624, amax_19);  convert_element_type_624 = amax_19 = None
        exp_19: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_39: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_625: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_39, torch.bfloat16);  div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_431: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_625, [-1, 512, 512]);  convert_element_type_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_426: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_610, [4096, 1536])
        permute_213: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
        addmm_116: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg315_1, view_426, permute_213);  arg315_1 = view_426 = permute_213 = None
        view_427: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_116, [8, 512, 1536]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_428: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_427, [8, 512, 24, -1]);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_214: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        clone_136: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_429: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [-1, 512, 64]);  clone_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_39: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_431, view_429);  view_431 = view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_432: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [-1, 24, 512, 64]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_216: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_138: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_433: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [8, 512, -1]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_434: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_433, [4096, 1536]);  view_433 = None
        permute_217: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
        addmm_117: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg317_1, view_434, permute_217);  arg317_1 = view_434 = permute_217 = None
        view_435: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_117, [8, 512, 1536]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_136: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_435, convert_element_type_610);  view_435 = convert_element_type_610 = None
        convert_element_type_631: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.float32);  add_136 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_631, [2], correction = 0, keepdim = True)
        getitem_78: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        sub_59: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_631, getitem_79);  convert_element_type_631 = getitem_79 = None
        add_137: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-07);  getitem_78 = None
        rsqrt_39: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        mul_157: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
        mul_158: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, arg318_1);  mul_157 = arg318_1 = None
        add_138: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, arg319_1);  mul_158 = arg319_1 = None
        convert_element_type_632: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16);  add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_436: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_632, [4096, 1536])
        permute_218: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
        addmm_118: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg321_1, view_436, permute_218);  arg321_1 = view_436 = permute_218 = None
        view_437: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_118, [8, 512, 6144]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_636: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        mul_159: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_636, 0.5)
        mul_160: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_636, 0.7071067811865476);  convert_element_type_636 = None
        erf_19: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_139: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_161: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_159, add_139);  mul_159 = add_139 = None
        convert_element_type_637: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_438: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_637, [4096, 6144]);  convert_element_type_637 = None
        permute_219: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        addmm_119: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg323_1, view_438, permute_219);  arg323_1 = view_438 = permute_219 = None
        view_439: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_119, [8, 512, 1536]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_140: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_439, convert_element_type_632);  view_439 = convert_element_type_632 = None
        convert_element_type_641: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_140, torch.float32);  add_140 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_641, [2], correction = 0, keepdim = True)
        getitem_80: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant20: "f32[][]cpu" = self._tensor_constant20
        lift_fresh_copy_20: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
        mul_164: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_20, 1);  lift_fresh_copy_20 = None
        sqrt_20: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_164);  mul_164 = sqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_655: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_62: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_60: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_641, getitem_81);  convert_element_type_641 = getitem_81 = None
        add_141: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-07);  getitem_80 = None
        rsqrt_40: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_162: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
        mul_163: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, arg324_1);  mul_162 = arg324_1 = None
        add_142: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_163, arg325_1);  mul_163 = arg325_1 = None
        convert_element_type_642: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_440: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_642, [4096, 1536])
        permute_220: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_120: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg327_1, view_440, permute_220);  arg327_1 = view_440 = permute_220 = None
        view_441: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_120, [8, 512, 1536]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_442: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [8, 512, 24, -1]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_221: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        clone_141: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_443: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_141, [-1, 512, 64]);  clone_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_444: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_642, [4096, 1536])
        permute_222: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_121: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg329_1, view_444, permute_222);  arg329_1 = view_444 = permute_222 = None
        view_445: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_121, [8, 512, 1536]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_446: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_445, [8, 512, 24, -1]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_223: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
        clone_142: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_447: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [-1, 512, 64]);  clone_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_226: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None
        full_default_61: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_40: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_226, full_default_61);  permute_226 = full_default_61 = None
        bmm_40: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_443, div_40);  view_443 = div_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_452: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [-1, 24, 512, 512]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_20: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_62, full_default_63, view_452);  full_default_62 = full_default_63 = view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_656: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        amax_20: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_656, [-1], True)
        sub_61: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_656, amax_20);  convert_element_type_656 = amax_20 = None
        exp_20: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_41: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_657: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_41, torch.bfloat16);  div_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_453: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_657, [-1, 512, 512]);  convert_element_type_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_448: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_642, [4096, 1536])
        permute_224: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        addmm_122: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg331_1, view_448, permute_224);  arg331_1 = view_448 = permute_224 = None
        view_449: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_122, [8, 512, 1536]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_450: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_449, [8, 512, 24, -1]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_225: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
        clone_143: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_451: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_143, [-1, 512, 64]);  clone_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_41: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_453, view_451);  view_453 = view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_454: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [-1, 24, 512, 64]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_227: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_145: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_455: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [8, 512, -1]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_456: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_455, [4096, 1536]);  view_455 = None
        permute_228: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        addmm_123: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg333_1, view_456, permute_228);  arg333_1 = view_456 = permute_228 = None
        view_457: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_123, [8, 512, 1536]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_143: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_457, convert_element_type_642);  view_457 = convert_element_type_642 = None
        convert_element_type_663: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.float32);  add_143 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_663, [2], correction = 0, keepdim = True)
        getitem_82: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        sub_62: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_663, getitem_83);  convert_element_type_663 = getitem_83 = None
        add_144: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-07);  getitem_82 = None
        rsqrt_41: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_165: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
        mul_166: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, arg334_1);  mul_165 = arg334_1 = None
        add_145: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_166, arg335_1);  mul_166 = arg335_1 = None
        convert_element_type_664: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_458: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_664, [4096, 1536])
        permute_229: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
        addmm_124: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg337_1, view_458, permute_229);  arg337_1 = view_458 = permute_229 = None
        view_459: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_124, [8, 512, 6144]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_668: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        mul_167: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_668, 0.5)
        mul_168: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_668, 0.7071067811865476);  convert_element_type_668 = None
        erf_20: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_146: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_169: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, add_146);  mul_167 = add_146 = None
        convert_element_type_669: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_460: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_669, [4096, 6144]);  convert_element_type_669 = None
        permute_230: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        addmm_125: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg339_1, view_460, permute_230);  arg339_1 = view_460 = permute_230 = None
        view_461: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_125, [8, 512, 1536]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_147: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_461, convert_element_type_664);  view_461 = convert_element_type_664 = None
        convert_element_type_673: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.float32);  add_147 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_673, [2], correction = 0, keepdim = True)
        getitem_84: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant21: "f32[][]cpu" = self._tensor_constant21
        lift_fresh_copy_21: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant21);  _tensor_constant21 = None
        mul_172: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_21, 1);  lift_fresh_copy_21 = None
        sqrt_21: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_172);  mul_172 = sqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_687: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_65: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_63: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_673, getitem_85);  convert_element_type_673 = getitem_85 = None
        add_148: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-07);  getitem_84 = None
        rsqrt_42: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_170: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
        mul_171: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, arg340_1);  mul_170 = arg340_1 = None
        add_149: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_171, arg341_1);  mul_171 = arg341_1 = None
        convert_element_type_674: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_462: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_674, [4096, 1536])
        permute_231: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
        addmm_126: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg343_1, view_462, permute_231);  arg343_1 = view_462 = permute_231 = None
        view_463: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_126, [8, 512, 1536]);  addmm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_464: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_463, [8, 512, 24, -1]);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_232: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
        clone_148: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_465: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_148, [-1, 512, 64]);  clone_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_466: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_674, [4096, 1536])
        permute_233: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        addmm_127: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg345_1, view_466, permute_233);  arg345_1 = view_466 = permute_233 = None
        view_467: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_127, [8, 512, 1536]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_468: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_467, [8, 512, 24, -1]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_234: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        clone_149: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_469: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_149, [-1, 512, 64]);  clone_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_237: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
        full_default_64: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_42: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_237, full_default_64);  permute_237 = full_default_64 = None
        bmm_42: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_465, div_42);  view_465 = div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_474: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [-1, 24, 512, 512]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_21: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_65, full_default_66, view_474);  full_default_65 = full_default_66 = view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_688: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        amax_21: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_688, [-1], True)
        sub_64: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_688, amax_21);  convert_element_type_688 = amax_21 = None
        exp_21: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_43: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_689: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16);  div_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_475: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_689, [-1, 512, 512]);  convert_element_type_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_470: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_674, [4096, 1536])
        permute_235: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
        addmm_128: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg347_1, view_470, permute_235);  arg347_1 = view_470 = permute_235 = None
        view_471: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_128, [8, 512, 1536]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_472: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_471, [8, 512, 24, -1]);  view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_236: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        clone_150: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_473: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_150, [-1, 512, 64]);  clone_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_43: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_475, view_473);  view_475 = view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_476: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [-1, 24, 512, 64]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_238: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_152: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_477: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_152, [8, 512, -1]);  clone_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_478: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_477, [4096, 1536]);  view_477 = None
        permute_239: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg348_1, [1, 0]);  arg348_1 = None
        addmm_129: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg349_1, view_478, permute_239);  arg349_1 = view_478 = permute_239 = None
        view_479: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_129, [8, 512, 1536]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_150: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_479, convert_element_type_674);  view_479 = convert_element_type_674 = None
        convert_element_type_695: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_150, torch.float32);  add_150 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_695, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        sub_65: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_695, getitem_87);  convert_element_type_695 = getitem_87 = None
        add_151: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-07);  getitem_86 = None
        rsqrt_43: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_173: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
        mul_174: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, arg350_1);  mul_173 = arg350_1 = None
        add_152: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_174, arg351_1);  mul_174 = arg351_1 = None
        convert_element_type_696: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_480: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_696, [4096, 1536])
        permute_240: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg352_1, [1, 0]);  arg352_1 = None
        addmm_130: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg353_1, view_480, permute_240);  arg353_1 = view_480 = permute_240 = None
        view_481: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_130, [8, 512, 6144]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_700: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_481, torch.float32);  view_481 = None
        mul_175: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_700, 0.5)
        mul_176: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_700, 0.7071067811865476);  convert_element_type_700 = None
        erf_21: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_176);  mul_176 = None
        add_153: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_177: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, add_153);  mul_175 = add_153 = None
        convert_element_type_701: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_177, torch.bfloat16);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_482: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_701, [4096, 6144]);  convert_element_type_701 = None
        permute_241: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
        addmm_131: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg355_1, view_482, permute_241);  arg355_1 = view_482 = permute_241 = None
        view_483: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_131, [8, 512, 1536]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_154: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_483, convert_element_type_696);  view_483 = convert_element_type_696 = None
        convert_element_type_705: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.float32);  add_154 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_705, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant22: "f32[][]cpu" = self._tensor_constant22
        lift_fresh_copy_22: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
        mul_180: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_22, 1);  lift_fresh_copy_22 = None
        sqrt_22: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_180);  mul_180 = sqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_719: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_68: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_69: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_66: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_705, getitem_89);  convert_element_type_705 = getitem_89 = None
        add_155: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-07);  getitem_88 = None
        rsqrt_44: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_178: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
        mul_179: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, arg356_1);  mul_178 = arg356_1 = None
        add_156: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, arg357_1);  mul_179 = arg357_1 = None
        convert_element_type_706: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_156, torch.bfloat16);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_484: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_706, [4096, 1536])
        permute_242: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
        addmm_132: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg359_1, view_484, permute_242);  arg359_1 = view_484 = permute_242 = None
        view_485: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_132, [8, 512, 1536]);  addmm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_486: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [8, 512, 24, -1]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_243: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        clone_155: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_487: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_155, [-1, 512, 64]);  clone_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_488: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_706, [4096, 1536])
        permute_244: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
        addmm_133: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg361_1, view_488, permute_244);  arg361_1 = view_488 = permute_244 = None
        view_489: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_133, [8, 512, 1536]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_490: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_489, [8, 512, 24, -1]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_245: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
        clone_156: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
        view_491: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_156, [-1, 512, 64]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_248: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_491, [0, 2, 1]);  view_491 = None
        full_default_67: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_44: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_248, full_default_67);  permute_248 = full_default_67 = None
        bmm_44: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_487, div_44);  view_487 = div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_496: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [-1, 24, 512, 512]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_22: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_68, full_default_69, view_496);  full_default_68 = full_default_69 = view_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_720: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        amax_22: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_720, [-1], True)
        sub_67: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_720, amax_22);  convert_element_type_720 = amax_22 = None
        exp_22: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_45: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_721: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_45, torch.bfloat16);  div_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_497: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_721, [-1, 512, 512]);  convert_element_type_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_492: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_706, [4096, 1536])
        permute_246: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
        addmm_134: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg363_1, view_492, permute_246);  arg363_1 = view_492 = permute_246 = None
        view_493: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_134, [8, 512, 1536]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_494: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_493, [8, 512, 24, -1]);  view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_247: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
        clone_157: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_495: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_157, [-1, 512, 64]);  clone_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_45: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_497, view_495);  view_497 = view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_498: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [-1, 24, 512, 64]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_249: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_159: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_499: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_159, [8, 512, -1]);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_500: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_499, [4096, 1536]);  view_499 = None
        permute_250: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
        addmm_135: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg365_1, view_500, permute_250);  arg365_1 = view_500 = permute_250 = None
        view_501: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_135, [8, 512, 1536]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_157: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_501, convert_element_type_706);  view_501 = convert_element_type_706 = None
        convert_element_type_727: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.float32);  add_157 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_727, [2], correction = 0, keepdim = True)
        getitem_90: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        sub_68: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_727, getitem_91);  convert_element_type_727 = getitem_91 = None
        add_158: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-07);  getitem_90 = None
        rsqrt_45: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        mul_181: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
        mul_182: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, arg366_1);  mul_181 = arg366_1 = None
        add_159: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_182, arg367_1);  mul_182 = arg367_1 = None
        convert_element_type_728: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_502: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_728, [4096, 1536])
        permute_251: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg368_1, [1, 0]);  arg368_1 = None
        addmm_136: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg369_1, view_502, permute_251);  arg369_1 = view_502 = permute_251 = None
        view_503: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_136, [8, 512, 6144]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_732: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        mul_183: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_732, 0.5)
        mul_184: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_732, 0.7071067811865476);  convert_element_type_732 = None
        erf_22: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_184);  mul_184 = None
        add_160: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_185: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, add_160);  mul_183 = add_160 = None
        convert_element_type_733: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_733, [4096, 6144]);  convert_element_type_733 = None
        permute_252: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
        addmm_137: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg371_1, view_504, permute_252);  arg371_1 = view_504 = permute_252 = None
        view_505: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_137, [8, 512, 1536]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_161: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_505, convert_element_type_728);  view_505 = convert_element_type_728 = None
        convert_element_type_737: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.float32);  add_161 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_737, [2], correction = 0, keepdim = True)
        getitem_92: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_46[0]
        getitem_93: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        _tensor_constant23: "f32[][]cpu" = self._tensor_constant23
        lift_fresh_copy_23: "f32[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
        mul_188: "f32[][]cpu" = torch.ops.aten.mul.Tensor(lift_fresh_copy_23, 1);  lift_fresh_copy_23 = None
        sqrt_23: "f32[][]cpu" = torch.ops.aten.sqrt.default(mul_188);  mul_188 = sqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_751: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  mul_3 = convert_element_type_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_71: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_72: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_69: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_737, getitem_93);  convert_element_type_737 = getitem_93 = None
        add_162: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-07);  getitem_92 = None
        rsqrt_46: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_186: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
        mul_187: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, arg372_1);  mul_186 = arg372_1 = None
        add_163: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_187, arg373_1);  mul_187 = arg373_1 = None
        convert_element_type_738: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_163, torch.bfloat16);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_506: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_738, [4096, 1536])
        permute_253: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg374_1, [1, 0]);  arg374_1 = None
        addmm_138: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg375_1, view_506, permute_253);  arg375_1 = view_506 = permute_253 = None
        view_507: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_138, [8, 512, 1536]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_508: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_507, [8, 512, 24, -1]);  view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_254: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        clone_162: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_509: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [-1, 512, 64]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        view_510: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_738, [4096, 1536])
        permute_255: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
        addmm_139: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg377_1, view_510, permute_255);  arg377_1 = view_510 = permute_255 = None
        view_511: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_139, [8, 512, 1536]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_512: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [8, 512, 24, -1]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_256: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_163: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_513: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_163, [-1, 512, 64]);  clone_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_259: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_513, [0, 2, 1]);  view_513 = None
        full_default_70: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_46: "bf16[192, 64, 512][32768, 1, 64]cuda:0" = torch.ops.aten.div.Tensor(permute_259, full_default_70);  permute_259 = full_default_70 = None
        bmm_46: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_509, div_46);  view_509 = div_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_518: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [-1, 24, 512, 512]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_23: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_71, full_default_72, view_518);  full_default_71 = full_default_72 = view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_752: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        amax_23: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_752, [-1], True)
        sub_70: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_752, amax_23);  convert_element_type_752 = amax_23 = None
        exp_23: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_47: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_753: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_47, torch.bfloat16);  div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_519: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_753, [-1, 512, 512]);  convert_element_type_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_514: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_738, [4096, 1536])
        permute_257: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        addmm_140: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg379_1, view_514, permute_257);  arg379_1 = view_514 = permute_257 = None
        view_515: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_140, [8, 512, 1536]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_516: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_515, [8, 512, 24, -1]);  view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_258: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
        clone_164: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_517: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_164, [-1, 512, 64]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_47: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_519, view_517);  view_519 = view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_520: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [-1, 24, 512, 64]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_260: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_166: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_521: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_166, [8, 512, -1]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_522: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_521, [4096, 1536]);  view_521 = None
        permute_261: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        addmm_141: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg381_1, view_522, permute_261);  arg381_1 = view_522 = permute_261 = None
        view_523: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_141, [8, 512, 1536]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_164: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_523, convert_element_type_738);  view_523 = convert_element_type_738 = None
        convert_element_type_759: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.float32);  add_164 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_759, [2], correction = 0, keepdim = True)
        getitem_94: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_47[0]
        getitem_95: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        sub_71: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_759, getitem_95);  convert_element_type_759 = getitem_95 = None
        add_165: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-07);  getitem_94 = None
        rsqrt_47: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        mul_189: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
        mul_190: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, arg382_1);  mul_189 = arg382_1 = None
        add_166: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_190, arg383_1);  mul_190 = arg383_1 = None
        convert_element_type_760: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_166, torch.bfloat16);  add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_524: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_760, [4096, 1536])
        permute_262: "bf16[1536, 6144][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg384_1, [1, 0]);  arg384_1 = None
        addmm_142: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.addmm.default(arg385_1, view_524, permute_262);  arg385_1 = view_524 = permute_262 = None
        view_525: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_142, [8, 512, 6144]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_764: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_525, torch.float32);  view_525 = None
        mul_191: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_764, 0.5)
        mul_192: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_764, 0.7071067811865476);  convert_element_type_764 = None
        erf_23: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_192);  mul_192 = None
        add_167: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_193: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_191, add_167);  mul_191 = add_167 = None
        convert_element_type_765: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_526: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_765, [4096, 6144]);  convert_element_type_765 = None
        permute_263: "bf16[6144, 1536][1, 6144]cuda:0" = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
        addmm_143: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg387_1, view_526, permute_263);  arg387_1 = view_526 = permute_263 = None
        view_527: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_143, [8, 512, 1536]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_168: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(view_527, convert_element_type_760);  view_527 = convert_element_type_760 = None
        convert_element_type_769: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_168, torch.float32);  add_168 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_769, [2], correction = 0, keepdim = True)
        getitem_96: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_48[0]
        getitem_97: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        sub_72: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_769, getitem_97);  convert_element_type_769 = getitem_97 = None
        add_169: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-07);  getitem_96 = None
        rsqrt_48: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        mul_194: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
        mul_195: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_194, arg388_1);  mul_194 = arg388_1 = None
        add_170: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, arg389_1);  mul_195 = arg389_1 = None
        convert_element_type_770: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16);  add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:818 in forward, code: hidden_states = self.dense(hidden_states)
        view_528: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_770, [4096, 1536]);  convert_element_type_770 = None
        permute_264: "bf16[1536, 1536][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg390_1, [1, 0]);  arg390_1 = None
        addmm_144: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg391_1, view_528, permute_264);  arg391_1 = view_528 = permute_264 = None
        view_529: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_144, [8, 512, 1536]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_774: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        mul_196: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_774, 0.5)
        mul_197: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_774, 0.7071067811865476);  convert_element_type_774 = None
        erf_24: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
        add_171: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_198: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, add_171);  mul_196 = add_171 = None
        convert_element_type_775: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_198, torch.bfloat16);  mul_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:820 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_776: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_775, torch.float32);  convert_element_type_775 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_776, [2], correction = 0, keepdim = True)
        getitem_98: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_49[0]
        getitem_99: "f32[8, 512, 1][512, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_533: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(arg395_1, [-1]);  arg395_1 = None
        ne_1: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_533, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        full_default_75: "bf16[4][1]cuda:0" = torch.ops.aten.full.default([4], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "bf16[128104][1]cuda:0" = torch.ops.aten.cat.default([arg394_1, full_default_75]);  arg394_1 = full_default_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:820 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_73: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_776, getitem_99);  convert_element_type_776 = getitem_99 = None
        add_172: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-07);  getitem_98 = None
        rsqrt_49: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        mul_199: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = rsqrt_49 = None
        mul_200: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, arg392_1);  mul_199 = arg392_1 = None
        add_173: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_200, arg393_1);  mul_200 = arg393_1 = None
        convert_element_type_777: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        view_530: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_777, [4096, 1536]);  convert_element_type_777 = None
        permute_265: "bf16[1536, 128100][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        constant_pad_nd_default: "bf16[1536, 128104][128104, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_265, [0, 4, 0, 0]);  permute_265 = None
        addmm_default: "bf16[4096, 128104][128104, 1]cuda:0" = torch.ops.aten.addmm.default(cat_default, view_530, constant_pad_nd_default);  cat_default = view_530 = constant_pad_nd_default = None
        slice_tensor: "bf16[4096, 128100][128104, 1]cuda:0" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -4);  addmm_default = None
        view_531: "bf16[8, 512, 128100][65589248, 128104, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [8, 512, 128100]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_532: "bf16[4096, 128100][128104, 1]cuda:0" = torch.ops.aten.reshape.default(view_531, [-1, 128100])
        convert_element_type_781: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_532, torch.float32);  view_532 = None
        amax_24: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_781, [1], True)
        sub_74: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_781, amax_24);  convert_element_type_781 = amax_24 = None
        exp_24: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.exp.default(sub_74)
        sum_25: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_75: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, log);  sub_74 = log = None
        convert_element_type_782: "bf16[4096, 128100][128100, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_75, torch.bfloat16);  sub_75 = None
        ne: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_533, -100)
        full_default_73: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[4096][1]cuda:0" = torch.ops.aten.where.self(ne, view_533, full_default_73);  ne = full_default_73 = None
        unsqueeze_4: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather: "bf16[4096, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_782, 1, unsqueeze_4);  convert_element_type_782 = unsqueeze_4 = None
        squeeze_1: "bf16[4096][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[4096][1]cuda:0" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        full_default_74: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "bf16[4096][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_74);  ne_1 = neg = full_default_74 = None
        sum_27: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_25);  where_25 = None
        ne_2: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_533, -100);  view_533 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_783: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.bfloat16);  sum_26 = None
        div_48: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_783);  sum_27 = convert_element_type_783 = None
        return (div_48, view_531)

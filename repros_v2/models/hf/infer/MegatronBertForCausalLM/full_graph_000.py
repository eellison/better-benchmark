class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 512][512, 1]cuda:0", arg1_1: "i64[16, 512][512, 1]cuda:0", arg2_1: "bf16[29056, 1024][1024, 1]cuda:0", arg3_1: "i64[1, 512][512, 1]cuda:0", arg4_1: "bf16[2, 1024][1024, 1]cuda:0", arg5_1: "bf16[512, 1024][1024, 1]cuda:0", arg6_1: "bf16[1024][1]cuda:0", arg7_1: "bf16[1024][1]cuda:0", arg8_1: "bf16[1024, 1024][1024, 1]cuda:0", arg9_1: "bf16[1024][1]cuda:0", arg10_1: "bf16[1024, 1024][1024, 1]cuda:0", arg11_1: "bf16[1024][1]cuda:0", arg12_1: "bf16[1024, 1024][1024, 1]cuda:0", arg13_1: "bf16[1024][1]cuda:0", arg14_1: "bf16[1024, 1024][1024, 1]cuda:0", arg15_1: "bf16[1024][1]cuda:0", arg16_1: "bf16[1024][1]cuda:0", arg17_1: "bf16[1024][1]cuda:0", arg18_1: "bf16[4096, 1024][1024, 1]cuda:0", arg19_1: "bf16[4096][1]cuda:0", arg20_1: "bf16[1024, 4096][4096, 1]cuda:0", arg21_1: "bf16[1024][1]cuda:0", arg22_1: "bf16[1024][1]cuda:0", arg23_1: "bf16[1024][1]cuda:0", arg24_1: "bf16[1024, 1024][1024, 1]cuda:0", arg25_1: "bf16[1024][1]cuda:0", arg26_1: "bf16[1024, 1024][1024, 1]cuda:0", arg27_1: "bf16[1024][1]cuda:0", arg28_1: "bf16[1024, 1024][1024, 1]cuda:0", arg29_1: "bf16[1024][1]cuda:0", arg30_1: "bf16[1024, 1024][1024, 1]cuda:0", arg31_1: "bf16[1024][1]cuda:0", arg32_1: "bf16[1024][1]cuda:0", arg33_1: "bf16[1024][1]cuda:0", arg34_1: "bf16[4096, 1024][1024, 1]cuda:0", arg35_1: "bf16[4096][1]cuda:0", arg36_1: "bf16[1024, 4096][4096, 1]cuda:0", arg37_1: "bf16[1024][1]cuda:0", arg38_1: "bf16[1024][1]cuda:0", arg39_1: "bf16[1024][1]cuda:0", arg40_1: "bf16[1024, 1024][1024, 1]cuda:0", arg41_1: "bf16[1024][1]cuda:0", arg42_1: "bf16[1024, 1024][1024, 1]cuda:0", arg43_1: "bf16[1024][1]cuda:0", arg44_1: "bf16[1024, 1024][1024, 1]cuda:0", arg45_1: "bf16[1024][1]cuda:0", arg46_1: "bf16[1024, 1024][1024, 1]cuda:0", arg47_1: "bf16[1024][1]cuda:0", arg48_1: "bf16[1024][1]cuda:0", arg49_1: "bf16[1024][1]cuda:0", arg50_1: "bf16[4096, 1024][1024, 1]cuda:0", arg51_1: "bf16[4096][1]cuda:0", arg52_1: "bf16[1024, 4096][4096, 1]cuda:0", arg53_1: "bf16[1024][1]cuda:0", arg54_1: "bf16[1024][1]cuda:0", arg55_1: "bf16[1024][1]cuda:0", arg56_1: "bf16[1024, 1024][1024, 1]cuda:0", arg57_1: "bf16[1024][1]cuda:0", arg58_1: "bf16[1024, 1024][1024, 1]cuda:0", arg59_1: "bf16[1024][1]cuda:0", arg60_1: "bf16[1024, 1024][1024, 1]cuda:0", arg61_1: "bf16[1024][1]cuda:0", arg62_1: "bf16[1024, 1024][1024, 1]cuda:0", arg63_1: "bf16[1024][1]cuda:0", arg64_1: "bf16[1024][1]cuda:0", arg65_1: "bf16[1024][1]cuda:0", arg66_1: "bf16[4096, 1024][1024, 1]cuda:0", arg67_1: "bf16[4096][1]cuda:0", arg68_1: "bf16[1024, 4096][4096, 1]cuda:0", arg69_1: "bf16[1024][1]cuda:0", arg70_1: "bf16[1024][1]cuda:0", arg71_1: "bf16[1024][1]cuda:0", arg72_1: "bf16[1024, 1024][1024, 1]cuda:0", arg73_1: "bf16[1024][1]cuda:0", arg74_1: "bf16[1024, 1024][1024, 1]cuda:0", arg75_1: "bf16[1024][1]cuda:0", arg76_1: "bf16[1024, 1024][1024, 1]cuda:0", arg77_1: "bf16[1024][1]cuda:0", arg78_1: "bf16[1024, 1024][1024, 1]cuda:0", arg79_1: "bf16[1024][1]cuda:0", arg80_1: "bf16[1024][1]cuda:0", arg81_1: "bf16[1024][1]cuda:0", arg82_1: "bf16[4096, 1024][1024, 1]cuda:0", arg83_1: "bf16[4096][1]cuda:0", arg84_1: "bf16[1024, 4096][4096, 1]cuda:0", arg85_1: "bf16[1024][1]cuda:0", arg86_1: "bf16[1024][1]cuda:0", arg87_1: "bf16[1024][1]cuda:0", arg88_1: "bf16[1024, 1024][1024, 1]cuda:0", arg89_1: "bf16[1024][1]cuda:0", arg90_1: "bf16[1024, 1024][1024, 1]cuda:0", arg91_1: "bf16[1024][1]cuda:0", arg92_1: "bf16[1024, 1024][1024, 1]cuda:0", arg93_1: "bf16[1024][1]cuda:0", arg94_1: "bf16[1024, 1024][1024, 1]cuda:0", arg95_1: "bf16[1024][1]cuda:0", arg96_1: "bf16[1024][1]cuda:0", arg97_1: "bf16[1024][1]cuda:0", arg98_1: "bf16[4096, 1024][1024, 1]cuda:0", arg99_1: "bf16[4096][1]cuda:0", arg100_1: "bf16[1024, 4096][4096, 1]cuda:0", arg101_1: "bf16[1024][1]cuda:0", arg102_1: "bf16[1024][1]cuda:0", arg103_1: "bf16[1024][1]cuda:0", arg104_1: "bf16[1024, 1024][1024, 1]cuda:0", arg105_1: "bf16[1024][1]cuda:0", arg106_1: "bf16[1024, 1024][1024, 1]cuda:0", arg107_1: "bf16[1024][1]cuda:0", arg108_1: "bf16[1024, 1024][1024, 1]cuda:0", arg109_1: "bf16[1024][1]cuda:0", arg110_1: "bf16[1024, 1024][1024, 1]cuda:0", arg111_1: "bf16[1024][1]cuda:0", arg112_1: "bf16[1024][1]cuda:0", arg113_1: "bf16[1024][1]cuda:0", arg114_1: "bf16[4096, 1024][1024, 1]cuda:0", arg115_1: "bf16[4096][1]cuda:0", arg116_1: "bf16[1024, 4096][4096, 1]cuda:0", arg117_1: "bf16[1024][1]cuda:0", arg118_1: "bf16[1024][1]cuda:0", arg119_1: "bf16[1024][1]cuda:0", arg120_1: "bf16[1024, 1024][1024, 1]cuda:0", arg121_1: "bf16[1024][1]cuda:0", arg122_1: "bf16[1024, 1024][1024, 1]cuda:0", arg123_1: "bf16[1024][1]cuda:0", arg124_1: "bf16[1024, 1024][1024, 1]cuda:0", arg125_1: "bf16[1024][1]cuda:0", arg126_1: "bf16[1024, 1024][1024, 1]cuda:0", arg127_1: "bf16[1024][1]cuda:0", arg128_1: "bf16[1024][1]cuda:0", arg129_1: "bf16[1024][1]cuda:0", arg130_1: "bf16[4096, 1024][1024, 1]cuda:0", arg131_1: "bf16[4096][1]cuda:0", arg132_1: "bf16[1024, 4096][4096, 1]cuda:0", arg133_1: "bf16[1024][1]cuda:0", arg134_1: "bf16[1024][1]cuda:0", arg135_1: "bf16[1024][1]cuda:0", arg136_1: "bf16[1024, 1024][1024, 1]cuda:0", arg137_1: "bf16[1024][1]cuda:0", arg138_1: "bf16[1024, 1024][1024, 1]cuda:0", arg139_1: "bf16[1024][1]cuda:0", arg140_1: "bf16[1024, 1024][1024, 1]cuda:0", arg141_1: "bf16[1024][1]cuda:0", arg142_1: "bf16[1024, 1024][1024, 1]cuda:0", arg143_1: "bf16[1024][1]cuda:0", arg144_1: "bf16[1024][1]cuda:0", arg145_1: "bf16[1024][1]cuda:0", arg146_1: "bf16[4096, 1024][1024, 1]cuda:0", arg147_1: "bf16[4096][1]cuda:0", arg148_1: "bf16[1024, 4096][4096, 1]cuda:0", arg149_1: "bf16[1024][1]cuda:0", arg150_1: "bf16[1024][1]cuda:0", arg151_1: "bf16[1024][1]cuda:0", arg152_1: "bf16[1024, 1024][1024, 1]cuda:0", arg153_1: "bf16[1024][1]cuda:0", arg154_1: "bf16[1024, 1024][1024, 1]cuda:0", arg155_1: "bf16[1024][1]cuda:0", arg156_1: "bf16[1024, 1024][1024, 1]cuda:0", arg157_1: "bf16[1024][1]cuda:0", arg158_1: "bf16[1024, 1024][1024, 1]cuda:0", arg159_1: "bf16[1024][1]cuda:0", arg160_1: "bf16[1024][1]cuda:0", arg161_1: "bf16[1024][1]cuda:0", arg162_1: "bf16[4096, 1024][1024, 1]cuda:0", arg163_1: "bf16[4096][1]cuda:0", arg164_1: "bf16[1024, 4096][4096, 1]cuda:0", arg165_1: "bf16[1024][1]cuda:0", arg166_1: "bf16[1024][1]cuda:0", arg167_1: "bf16[1024][1]cuda:0", arg168_1: "bf16[1024, 1024][1024, 1]cuda:0", arg169_1: "bf16[1024][1]cuda:0", arg170_1: "bf16[1024, 1024][1024, 1]cuda:0", arg171_1: "bf16[1024][1]cuda:0", arg172_1: "bf16[1024, 1024][1024, 1]cuda:0", arg173_1: "bf16[1024][1]cuda:0", arg174_1: "bf16[1024, 1024][1024, 1]cuda:0", arg175_1: "bf16[1024][1]cuda:0", arg176_1: "bf16[1024][1]cuda:0", arg177_1: "bf16[1024][1]cuda:0", arg178_1: "bf16[4096, 1024][1024, 1]cuda:0", arg179_1: "bf16[4096][1]cuda:0", arg180_1: "bf16[1024, 4096][4096, 1]cuda:0", arg181_1: "bf16[1024][1]cuda:0", arg182_1: "bf16[1024][1]cuda:0", arg183_1: "bf16[1024][1]cuda:0", arg184_1: "bf16[1024, 1024][1024, 1]cuda:0", arg185_1: "bf16[1024][1]cuda:0", arg186_1: "bf16[1024, 1024][1024, 1]cuda:0", arg187_1: "bf16[1024][1]cuda:0", arg188_1: "bf16[1024, 1024][1024, 1]cuda:0", arg189_1: "bf16[1024][1]cuda:0", arg190_1: "bf16[1024, 1024][1024, 1]cuda:0", arg191_1: "bf16[1024][1]cuda:0", arg192_1: "bf16[1024][1]cuda:0", arg193_1: "bf16[1024][1]cuda:0", arg194_1: "bf16[4096, 1024][1024, 1]cuda:0", arg195_1: "bf16[4096][1]cuda:0", arg196_1: "bf16[1024, 4096][4096, 1]cuda:0", arg197_1: "bf16[1024][1]cuda:0", arg198_1: "bf16[1024][1]cuda:0", arg199_1: "bf16[1024][1]cuda:0", arg200_1: "bf16[1024, 1024][1024, 1]cuda:0", arg201_1: "bf16[1024][1]cuda:0", arg202_1: "bf16[1024, 1024][1024, 1]cuda:0", arg203_1: "bf16[1024][1]cuda:0", arg204_1: "bf16[1024, 1024][1024, 1]cuda:0", arg205_1: "bf16[1024][1]cuda:0", arg206_1: "bf16[1024, 1024][1024, 1]cuda:0", arg207_1: "bf16[1024][1]cuda:0", arg208_1: "bf16[1024][1]cuda:0", arg209_1: "bf16[1024][1]cuda:0", arg210_1: "bf16[4096, 1024][1024, 1]cuda:0", arg211_1: "bf16[4096][1]cuda:0", arg212_1: "bf16[1024, 4096][4096, 1]cuda:0", arg213_1: "bf16[1024][1]cuda:0", arg214_1: "bf16[1024][1]cuda:0", arg215_1: "bf16[1024][1]cuda:0", arg216_1: "bf16[1024, 1024][1024, 1]cuda:0", arg217_1: "bf16[1024][1]cuda:0", arg218_1: "bf16[1024, 1024][1024, 1]cuda:0", arg219_1: "bf16[1024][1]cuda:0", arg220_1: "bf16[1024, 1024][1024, 1]cuda:0", arg221_1: "bf16[1024][1]cuda:0", arg222_1: "bf16[1024, 1024][1024, 1]cuda:0", arg223_1: "bf16[1024][1]cuda:0", arg224_1: "bf16[1024][1]cuda:0", arg225_1: "bf16[1024][1]cuda:0", arg226_1: "bf16[4096, 1024][1024, 1]cuda:0", arg227_1: "bf16[4096][1]cuda:0", arg228_1: "bf16[1024, 4096][4096, 1]cuda:0", arg229_1: "bf16[1024][1]cuda:0", arg230_1: "bf16[1024][1]cuda:0", arg231_1: "bf16[1024][1]cuda:0", arg232_1: "bf16[1024, 1024][1024, 1]cuda:0", arg233_1: "bf16[1024][1]cuda:0", arg234_1: "bf16[1024, 1024][1024, 1]cuda:0", arg235_1: "bf16[1024][1]cuda:0", arg236_1: "bf16[1024, 1024][1024, 1]cuda:0", arg237_1: "bf16[1024][1]cuda:0", arg238_1: "bf16[1024, 1024][1024, 1]cuda:0", arg239_1: "bf16[1024][1]cuda:0", arg240_1: "bf16[1024][1]cuda:0", arg241_1: "bf16[1024][1]cuda:0", arg242_1: "bf16[4096, 1024][1024, 1]cuda:0", arg243_1: "bf16[4096][1]cuda:0", arg244_1: "bf16[1024, 4096][4096, 1]cuda:0", arg245_1: "bf16[1024][1]cuda:0", arg246_1: "bf16[1024][1]cuda:0", arg247_1: "bf16[1024][1]cuda:0", arg248_1: "bf16[1024, 1024][1024, 1]cuda:0", arg249_1: "bf16[1024][1]cuda:0", arg250_1: "bf16[1024, 1024][1024, 1]cuda:0", arg251_1: "bf16[1024][1]cuda:0", arg252_1: "bf16[1024, 1024][1024, 1]cuda:0", arg253_1: "bf16[1024][1]cuda:0", arg254_1: "bf16[1024, 1024][1024, 1]cuda:0", arg255_1: "bf16[1024][1]cuda:0", arg256_1: "bf16[1024][1]cuda:0", arg257_1: "bf16[1024][1]cuda:0", arg258_1: "bf16[4096, 1024][1024, 1]cuda:0", arg259_1: "bf16[4096][1]cuda:0", arg260_1: "bf16[1024, 4096][4096, 1]cuda:0", arg261_1: "bf16[1024][1]cuda:0", arg262_1: "bf16[1024][1]cuda:0", arg263_1: "bf16[1024][1]cuda:0", arg264_1: "bf16[1024, 1024][1024, 1]cuda:0", arg265_1: "bf16[1024][1]cuda:0", arg266_1: "bf16[1024, 1024][1024, 1]cuda:0", arg267_1: "bf16[1024][1]cuda:0", arg268_1: "bf16[1024, 1024][1024, 1]cuda:0", arg269_1: "bf16[1024][1]cuda:0", arg270_1: "bf16[1024, 1024][1024, 1]cuda:0", arg271_1: "bf16[1024][1]cuda:0", arg272_1: "bf16[1024][1]cuda:0", arg273_1: "bf16[1024][1]cuda:0", arg274_1: "bf16[4096, 1024][1024, 1]cuda:0", arg275_1: "bf16[4096][1]cuda:0", arg276_1: "bf16[1024, 4096][4096, 1]cuda:0", arg277_1: "bf16[1024][1]cuda:0", arg278_1: "bf16[1024][1]cuda:0", arg279_1: "bf16[1024][1]cuda:0", arg280_1: "bf16[1024, 1024][1024, 1]cuda:0", arg281_1: "bf16[1024][1]cuda:0", arg282_1: "bf16[1024, 1024][1024, 1]cuda:0", arg283_1: "bf16[1024][1]cuda:0", arg284_1: "bf16[1024, 1024][1024, 1]cuda:0", arg285_1: "bf16[1024][1]cuda:0", arg286_1: "bf16[1024, 1024][1024, 1]cuda:0", arg287_1: "bf16[1024][1]cuda:0", arg288_1: "bf16[1024][1]cuda:0", arg289_1: "bf16[1024][1]cuda:0", arg290_1: "bf16[4096, 1024][1024, 1]cuda:0", arg291_1: "bf16[4096][1]cuda:0", arg292_1: "bf16[1024, 4096][4096, 1]cuda:0", arg293_1: "bf16[1024][1]cuda:0", arg294_1: "bf16[1024][1]cuda:0", arg295_1: "bf16[1024][1]cuda:0", arg296_1: "bf16[1024, 1024][1024, 1]cuda:0", arg297_1: "bf16[1024][1]cuda:0", arg298_1: "bf16[1024, 1024][1024, 1]cuda:0", arg299_1: "bf16[1024][1]cuda:0", arg300_1: "bf16[1024, 1024][1024, 1]cuda:0", arg301_1: "bf16[1024][1]cuda:0", arg302_1: "bf16[1024, 1024][1024, 1]cuda:0", arg303_1: "bf16[1024][1]cuda:0", arg304_1: "bf16[1024][1]cuda:0", arg305_1: "bf16[1024][1]cuda:0", arg306_1: "bf16[4096, 1024][1024, 1]cuda:0", arg307_1: "bf16[4096][1]cuda:0", arg308_1: "bf16[1024, 4096][4096, 1]cuda:0", arg309_1: "bf16[1024][1]cuda:0", arg310_1: "bf16[1024][1]cuda:0", arg311_1: "bf16[1024][1]cuda:0", arg312_1: "bf16[1024, 1024][1024, 1]cuda:0", arg313_1: "bf16[1024][1]cuda:0", arg314_1: "bf16[1024, 1024][1024, 1]cuda:0", arg315_1: "bf16[1024][1]cuda:0", arg316_1: "bf16[1024, 1024][1024, 1]cuda:0", arg317_1: "bf16[1024][1]cuda:0", arg318_1: "bf16[1024, 1024][1024, 1]cuda:0", arg319_1: "bf16[1024][1]cuda:0", arg320_1: "bf16[1024][1]cuda:0", arg321_1: "bf16[1024][1]cuda:0", arg322_1: "bf16[4096, 1024][1024, 1]cuda:0", arg323_1: "bf16[4096][1]cuda:0", arg324_1: "bf16[1024, 4096][4096, 1]cuda:0", arg325_1: "bf16[1024][1]cuda:0", arg326_1: "bf16[1024][1]cuda:0", arg327_1: "bf16[1024][1]cuda:0", arg328_1: "bf16[1024, 1024][1024, 1]cuda:0", arg329_1: "bf16[1024][1]cuda:0", arg330_1: "bf16[1024, 1024][1024, 1]cuda:0", arg331_1: "bf16[1024][1]cuda:0", arg332_1: "bf16[1024, 1024][1024, 1]cuda:0", arg333_1: "bf16[1024][1]cuda:0", arg334_1: "bf16[1024, 1024][1024, 1]cuda:0", arg335_1: "bf16[1024][1]cuda:0", arg336_1: "bf16[1024][1]cuda:0", arg337_1: "bf16[1024][1]cuda:0", arg338_1: "bf16[4096, 1024][1024, 1]cuda:0", arg339_1: "bf16[4096][1]cuda:0", arg340_1: "bf16[1024, 4096][4096, 1]cuda:0", arg341_1: "bf16[1024][1]cuda:0", arg342_1: "bf16[1024][1]cuda:0", arg343_1: "bf16[1024][1]cuda:0", arg344_1: "bf16[1024, 1024][1024, 1]cuda:0", arg345_1: "bf16[1024][1]cuda:0", arg346_1: "bf16[1024, 1024][1024, 1]cuda:0", arg347_1: "bf16[1024][1]cuda:0", arg348_1: "bf16[1024, 1024][1024, 1]cuda:0", arg349_1: "bf16[1024][1]cuda:0", arg350_1: "bf16[1024, 1024][1024, 1]cuda:0", arg351_1: "bf16[1024][1]cuda:0", arg352_1: "bf16[1024][1]cuda:0", arg353_1: "bf16[1024][1]cuda:0", arg354_1: "bf16[4096, 1024][1024, 1]cuda:0", arg355_1: "bf16[4096][1]cuda:0", arg356_1: "bf16[1024, 4096][4096, 1]cuda:0", arg357_1: "bf16[1024][1]cuda:0", arg358_1: "bf16[1024][1]cuda:0", arg359_1: "bf16[1024][1]cuda:0", arg360_1: "bf16[1024, 1024][1024, 1]cuda:0", arg361_1: "bf16[1024][1]cuda:0", arg362_1: "bf16[1024, 1024][1024, 1]cuda:0", arg363_1: "bf16[1024][1]cuda:0", arg364_1: "bf16[1024, 1024][1024, 1]cuda:0", arg365_1: "bf16[1024][1]cuda:0", arg366_1: "bf16[1024, 1024][1024, 1]cuda:0", arg367_1: "bf16[1024][1]cuda:0", arg368_1: "bf16[1024][1]cuda:0", arg369_1: "bf16[1024][1]cuda:0", arg370_1: "bf16[4096, 1024][1024, 1]cuda:0", arg371_1: "bf16[4096][1]cuda:0", arg372_1: "bf16[1024, 4096][4096, 1]cuda:0", arg373_1: "bf16[1024][1]cuda:0", arg374_1: "bf16[1024][1]cuda:0", arg375_1: "bf16[1024][1]cuda:0", arg376_1: "bf16[1024, 1024][1024, 1]cuda:0", arg377_1: "bf16[1024][1]cuda:0", arg378_1: "bf16[1024, 1024][1024, 1]cuda:0", arg379_1: "bf16[1024][1]cuda:0", arg380_1: "bf16[1024, 1024][1024, 1]cuda:0", arg381_1: "bf16[1024][1]cuda:0", arg382_1: "bf16[1024, 1024][1024, 1]cuda:0", arg383_1: "bf16[1024][1]cuda:0", arg384_1: "bf16[1024][1]cuda:0", arg385_1: "bf16[1024][1]cuda:0", arg386_1: "bf16[4096, 1024][1024, 1]cuda:0", arg387_1: "bf16[4096][1]cuda:0", arg388_1: "bf16[1024, 4096][4096, 1]cuda:0", arg389_1: "bf16[1024][1]cuda:0", arg390_1: "bf16[1024][1]cuda:0", arg391_1: "bf16[1024][1]cuda:0", arg392_1: "bf16[1024, 1024][1024, 1]cuda:0", arg393_1: "bf16[1024][1]cuda:0", arg394_1: "bf16[1024][1]cuda:0", arg395_1: "bf16[1024][1]cuda:0", arg396_1: "bf16[29056][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:626 in forward, code: attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        full: "f32[16, 512][512, 1]cuda:0" = torch.ops.aten.full.default([16, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:998 in get_extended_attention_mask, code: extended_attention_mask = attention_mask[:, None, None, :]
        unsqueeze: "f32[16, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1: "f32[16, 1, 1, 512][512, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1009 in get_extended_attention_mask, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        convert_element_type: "bf16[16, 1, 1, 512][512, 512, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.bfloat16);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1010 in get_extended_attention_mask, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        sub: "bf16[16, 1, 1, 512][512, 512, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, convert_element_type);  convert_element_type = sub = None
        full_default_1: "bf16[16, 1, 1, 512][512, 512, 512, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 1, 512], -0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:89 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 0);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:628 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[16, 512][512, 1]cuda:0" = torch.ops.aten.full.default([16, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:90 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg4_1, full_default);  arg4_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:91 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:93 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "bf16[1, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg5_1, arg3_1);  arg5_1 = arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:94 in forward, code: embeddings += position_embeddings
        add_1: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_1: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_1, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub_1: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = getitem_1 = None
        add_2: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_1: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_2: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        add_3: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
        convert_element_type_2: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [8192, 1024])
        permute: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [16, 512, 1024]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_2: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [16, 512, -1, 64]);  view_1 = None

        # No stacktrace found for following nodes
        permute_default_69: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_3: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [8192, 1024])
        permute_2: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_1: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg11_1, view_3, permute_2);  arg11_1 = view_3 = permute_2 = None
        view_4: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [16, 512, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_5: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [16, 512, -1, 64]);  view_4 = None

        # No stacktrace found for following nodes
        permute_default_70: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_6: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [8192, 1024]);  convert_element_type_2 = None
        permute_4: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg13_1, view_6, permute_4);  arg13_1 = view_6 = permute_4 = None
        view_7: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [16, 512, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_8: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_7, [16, 512, -1, 64]);  view_7 = None

        # No stacktrace found for following nodes
        permute_default_71: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        _scaled_dot_product_cudnn_attention_default_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_69, permute_default_70, permute_default_71, None, False, scale = 0.125);  permute_default_69 = permute_default_70 = permute_default_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_123: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_23[0];  _scaled_dot_product_cudnn_attention_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_7: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_123, [0, 2, 1, 3]);  getitem_123 = None
        clone_5: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_15: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [16, 512, 1024]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [8192, 1024]);  view_15 = None
        permute_8: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_3: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg15_1, view_16, permute_8);  arg15_1 = view_16 = permute_8 = None
        view_17: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [16, 512, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_5: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1, view_17);  add_1 = view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_21: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_21, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_3: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_21, getitem_3);  convert_element_type_21 = getitem_3 = None
        add_6: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_3: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_4: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, arg16_1);  mul_3 = arg16_1 = None
        add_7: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, arg17_1);  mul_4 = arg17_1 = None
        convert_element_type_22: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_22, [8192, 1024]);  convert_element_type_22 = None
        permute_9: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_4: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg19_1, view_18, permute_9);  arg19_1 = view_18 = permute_9 = None
        view_19: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [16, 512, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_26: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_5: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.5)
        mul_6: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476);  convert_element_type_26 = None
        erf: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_8: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
        convert_element_type_27: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_27, [8192, 4096]);  convert_element_type_27 = None
        permute_10: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_5: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg21_1, view_20, permute_10);  arg21_1 = view_20 = permute_10 = None
        view_21: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [16, 512, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_9: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_5, view_21);  add_5 = view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_31: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_31, [2], correction = 0, keepdim = True)
        getitem_4: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_4: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_31, getitem_5);  convert_element_type_31 = getitem_5 = None
        add_10: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_8: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, arg22_1);  mul_8 = arg22_1 = None
        add_11: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, arg23_1);  mul_9 = arg23_1 = None
        convert_element_type_32: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_22: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_32, [8192, 1024])
        permute_11: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_6: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg25_1, view_22, permute_11);  arg25_1 = view_22 = permute_11 = None
        view_23: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [16, 512, 1024]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_24: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [16, 512, -1, 64]);  view_23 = None

        # No stacktrace found for following nodes
        permute_default_66: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_25: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_32, [8192, 1024])
        permute_13: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_7: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg27_1, view_25, permute_13);  arg27_1 = view_25 = permute_13 = None
        view_26: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [16, 512, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_27: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [16, 512, -1, 64]);  view_26 = None

        # No stacktrace found for following nodes
        permute_default_67: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_28: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_32, [8192, 1024]);  convert_element_type_32 = None
        permute_15: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_8: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg29_1, view_28, permute_15);  arg29_1 = view_28 = permute_15 = None
        view_29: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [16, 512, 1024]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_30: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [16, 512, -1, 64]);  view_29 = None

        # No stacktrace found for following nodes
        permute_default_68: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        _scaled_dot_product_cudnn_attention_default_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_66, permute_default_67, permute_default_68, None, False, scale = 0.125);  permute_default_66 = permute_default_67 = permute_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_122: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_22[0];  _scaled_dot_product_cudnn_attention_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_18: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_122, [0, 2, 1, 3]);  getitem_122 = None
        clone_12: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_37: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [16, 512, 1024]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [8192, 1024]);  view_37 = None
        permute_19: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_9: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg31_1, view_38, permute_19);  arg31_1 = view_38 = permute_19 = None
        view_39: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [16, 512, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_13: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, view_39);  add_9 = view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_51: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_51, [2], correction = 0, keepdim = True)
        getitem_6: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_6: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_51, getitem_7);  convert_element_type_51 = getitem_7 = None
        add_14: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_10: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
        mul_11: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, arg32_1);  mul_10 = arg32_1 = None
        add_15: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, arg33_1);  mul_11 = arg33_1 = None
        convert_element_type_52: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_52, [8192, 1024]);  convert_element_type_52 = None
        permute_20: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_10: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg35_1, view_40, permute_20);  arg35_1 = view_40 = permute_20 = None
        view_41: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [16, 512, 4096]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_56: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_12: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, 0.5)
        mul_13: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, 0.7071067811865476);  convert_element_type_56 = None
        erf_1: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_16: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        convert_element_type_57: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_57, [8192, 4096]);  convert_element_type_57 = None
        permute_21: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_11: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg37_1, view_42, permute_21);  arg37_1 = view_42 = permute_21 = None
        view_43: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [16, 512, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_17: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, view_43);  add_13 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_61: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_61, [2], correction = 0, keepdim = True)
        getitem_8: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_7: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_61, getitem_9);  convert_element_type_61 = getitem_9 = None
        add_18: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_15: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_16: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, arg38_1);  mul_15 = arg38_1 = None
        add_19: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, arg39_1);  mul_16 = arg39_1 = None
        convert_element_type_62: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_44: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_62, [8192, 1024])
        permute_22: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_12: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg41_1, view_44, permute_22);  arg41_1 = view_44 = permute_22 = None
        view_45: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [16, 512, 1024]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_46: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [16, 512, -1, 64]);  view_45 = None

        # No stacktrace found for following nodes
        permute_default_63: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_47: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_62, [8192, 1024])
        permute_24: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_13: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg43_1, view_47, permute_24);  arg43_1 = view_47 = permute_24 = None
        view_48: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [16, 512, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_49: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [16, 512, -1, 64]);  view_48 = None

        # No stacktrace found for following nodes
        permute_default_64: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_50: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_62, [8192, 1024]);  convert_element_type_62 = None
        permute_26: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_14: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg45_1, view_50, permute_26);  arg45_1 = view_50 = permute_26 = None
        view_51: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [16, 512, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_52: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_51, [16, 512, -1, 64]);  view_51 = None

        # No stacktrace found for following nodes
        permute_default_65: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        _scaled_dot_product_cudnn_attention_default_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_63, permute_default_64, permute_default_65, None, False, scale = 0.125);  permute_default_63 = permute_default_64 = permute_default_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_121: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_21[0];  _scaled_dot_product_cudnn_attention_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_29: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3]);  getitem_121 = None
        clone_19: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_59: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [16, 512, 1024]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [8192, 1024]);  view_59 = None
        permute_30: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_15: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg47_1, view_60, permute_30);  arg47_1 = view_60 = permute_30 = None
        view_61: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [16, 512, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_21: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_17, view_61);  add_17 = view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_81: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_81, [2], correction = 0, keepdim = True)
        getitem_10: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_9: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_81, getitem_11);  convert_element_type_81 = getitem_11 = None
        add_22: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_17: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
        mul_18: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, arg48_1);  mul_17 = arg48_1 = None
        add_23: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, arg49_1);  mul_18 = arg49_1 = None
        convert_element_type_82: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_82, [8192, 1024]);  convert_element_type_82 = None
        permute_31: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_16: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg51_1, view_62, permute_31);  arg51_1 = view_62 = permute_31 = None
        view_63: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [16, 512, 4096]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_86: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_19: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.5)
        mul_20: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.7071067811865476);  convert_element_type_86 = None
        erf_2: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_24: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
        convert_element_type_87: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_87, [8192, 4096]);  convert_element_type_87 = None
        permute_32: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_17: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg53_1, view_64, permute_32);  arg53_1 = view_64 = permute_32 = None
        view_65: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [16, 512, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_25: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_21, view_65);  add_21 = view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_91: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_91, [2], correction = 0, keepdim = True)
        getitem_12: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_10: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_91, getitem_13);  convert_element_type_91 = getitem_13 = None
        add_26: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_22: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_23: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg54_1);  mul_22 = arg54_1 = None
        add_27: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg55_1);  mul_23 = arg55_1 = None
        convert_element_type_92: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_66: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [8192, 1024])
        permute_33: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg57_1, view_66, permute_33);  arg57_1 = view_66 = permute_33 = None
        view_67: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [16, 512, 1024]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_68: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [16, 512, -1, 64]);  view_67 = None

        # No stacktrace found for following nodes
        permute_default_60: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_69: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [8192, 1024])
        permute_35: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_19: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg59_1, view_69, permute_35);  arg59_1 = view_69 = permute_35 = None
        view_70: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [16, 512, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_71: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [16, 512, -1, 64]);  view_70 = None

        # No stacktrace found for following nodes
        permute_default_61: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_72: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [8192, 1024]);  convert_element_type_92 = None
        permute_37: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_20: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg61_1, view_72, permute_37);  arg61_1 = view_72 = permute_37 = None
        view_73: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [16, 512, 1024]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_74: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [16, 512, -1, 64]);  view_73 = None

        # No stacktrace found for following nodes
        permute_default_62: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        _scaled_dot_product_cudnn_attention_default_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_60, permute_default_61, permute_default_62, None, False, scale = 0.125);  permute_default_60 = permute_default_61 = permute_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_120: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_20[0];  _scaled_dot_product_cudnn_attention_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_40: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
        clone_26: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_81: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [16, 512, 1024]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [8192, 1024]);  view_81 = None
        permute_41: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_21: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg63_1, view_82, permute_41);  arg63_1 = view_82 = permute_41 = None
        view_83: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [16, 512, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_29: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_25, view_83);  add_25 = view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_111: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_111, [2], correction = 0, keepdim = True)
        getitem_14: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_12: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_111, getitem_15);  convert_element_type_111 = getitem_15 = None
        add_30: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_24: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
        mul_25: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, arg64_1);  mul_24 = arg64_1 = None
        add_31: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, arg65_1);  mul_25 = arg65_1 = None
        convert_element_type_112: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_112, [8192, 1024]);  convert_element_type_112 = None
        permute_42: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_22: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg67_1, view_84, permute_42);  arg67_1 = view_84 = permute_42 = None
        view_85: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [16, 512, 4096]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_116: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_26: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.5)
        mul_27: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.7071067811865476);  convert_element_type_116 = None
        erf_3: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_32: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
        convert_element_type_117: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_117, [8192, 4096]);  convert_element_type_117 = None
        permute_43: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_23: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg69_1, view_86, permute_43);  arg69_1 = view_86 = permute_43 = None
        view_87: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [16, 512, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_33: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_29, view_87);  add_29 = view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_121: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_121, [2], correction = 0, keepdim = True)
        getitem_16: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_13: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_121, getitem_17);  convert_element_type_121 = getitem_17 = None
        add_34: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_29: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_30: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, arg70_1);  mul_29 = arg70_1 = None
        add_35: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, arg71_1);  mul_30 = arg71_1 = None
        convert_element_type_122: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_88: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_122, [8192, 1024])
        permute_44: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_24: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg73_1, view_88, permute_44);  arg73_1 = view_88 = permute_44 = None
        view_89: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [16, 512, 1024]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_90: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [16, 512, -1, 64]);  view_89 = None

        # No stacktrace found for following nodes
        permute_default_57: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_91: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_122, [8192, 1024])
        permute_46: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_25: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg75_1, view_91, permute_46);  arg75_1 = view_91 = permute_46 = None
        view_92: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [16, 512, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_93: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [16, 512, -1, 64]);  view_92 = None

        # No stacktrace found for following nodes
        permute_default_58: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_94: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_122, [8192, 1024]);  convert_element_type_122 = None
        permute_48: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_26: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg77_1, view_94, permute_48);  arg77_1 = view_94 = permute_48 = None
        view_95: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [16, 512, 1024]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_96: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_95, [16, 512, -1, 64]);  view_95 = None

        # No stacktrace found for following nodes
        permute_default_59: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        _scaled_dot_product_cudnn_attention_default_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_57, permute_default_58, permute_default_59, None, False, scale = 0.125);  permute_default_57 = permute_default_58 = permute_default_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_119: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_19[0];  _scaled_dot_product_cudnn_attention_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_51: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None
        clone_33: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_103: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [16, 512, 1024]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [8192, 1024]);  view_103 = None
        permute_52: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_27: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg79_1, view_104, permute_52);  arg79_1 = view_104 = permute_52 = None
        view_105: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [16, 512, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_37: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_33, view_105);  add_33 = view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_141: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_141, [2], correction = 0, keepdim = True)
        getitem_18: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_15: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_141, getitem_19);  convert_element_type_141 = getitem_19 = None
        add_38: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_31: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
        mul_32: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, arg80_1);  mul_31 = arg80_1 = None
        add_39: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, arg81_1);  mul_32 = arg81_1 = None
        convert_element_type_142: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_142, [8192, 1024]);  convert_element_type_142 = None
        permute_53: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_28: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg83_1, view_106, permute_53);  arg83_1 = view_106 = permute_53 = None
        view_107: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [16, 512, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_146: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_33: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_146, 0.5)
        mul_34: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_146, 0.7071067811865476);  convert_element_type_146 = None
        erf_4: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_40: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
        convert_element_type_147: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_147, [8192, 4096]);  convert_element_type_147 = None
        permute_54: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_29: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg85_1, view_108, permute_54);  arg85_1 = view_108 = permute_54 = None
        view_109: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [16, 512, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_41: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, view_109);  add_37 = view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_151: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_151, [2], correction = 0, keepdim = True)
        getitem_20: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_16: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_151, getitem_21);  convert_element_type_151 = getitem_21 = None
        add_42: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_36: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_37: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg86_1);  mul_36 = arg86_1 = None
        add_43: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg87_1);  mul_37 = arg87_1 = None
        convert_element_type_152: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_110: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_152, [8192, 1024])
        permute_55: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_30: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg89_1, view_110, permute_55);  arg89_1 = view_110 = permute_55 = None
        view_111: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [16, 512, 1024]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_112: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [16, 512, -1, 64]);  view_111 = None

        # No stacktrace found for following nodes
        permute_default_54: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_113: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_152, [8192, 1024])
        permute_57: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_31: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg91_1, view_113, permute_57);  arg91_1 = view_113 = permute_57 = None
        view_114: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [16, 512, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_115: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [16, 512, -1, 64]);  view_114 = None

        # No stacktrace found for following nodes
        permute_default_55: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_116: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_152, [8192, 1024]);  convert_element_type_152 = None
        permute_59: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_32: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg93_1, view_116, permute_59);  arg93_1 = view_116 = permute_59 = None
        view_117: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [16, 512, 1024]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_118: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [16, 512, -1, 64]);  view_117 = None

        # No stacktrace found for following nodes
        permute_default_56: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        _scaled_dot_product_cudnn_attention_default_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_54, permute_default_55, permute_default_56, None, False, scale = 0.125);  permute_default_54 = permute_default_55 = permute_default_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_118: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_18[0];  _scaled_dot_product_cudnn_attention_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_62: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_118, [0, 2, 1, 3]);  getitem_118 = None
        clone_40: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_125: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [16, 512, 1024]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [8192, 1024]);  view_125 = None
        permute_63: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_33: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg95_1, view_126, permute_63);  arg95_1 = view_126 = permute_63 = None
        view_127: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [16, 512, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_45: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_41, view_127);  add_41 = view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_171: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_171, [2], correction = 0, keepdim = True)
        getitem_22: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_18: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_171, getitem_23);  convert_element_type_171 = getitem_23 = None
        add_46: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_38: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
        mul_39: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, arg96_1);  mul_38 = arg96_1 = None
        add_47: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, arg97_1);  mul_39 = arg97_1 = None
        convert_element_type_172: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_172, [8192, 1024]);  convert_element_type_172 = None
        permute_64: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_34: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg99_1, view_128, permute_64);  arg99_1 = view_128 = permute_64 = None
        view_129: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [16, 512, 4096]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_176: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_40: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_176, 0.5)
        mul_41: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_176, 0.7071067811865476);  convert_element_type_176 = None
        erf_5: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_48: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
        convert_element_type_177: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_177, [8192, 4096]);  convert_element_type_177 = None
        permute_65: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_35: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg101_1, view_130, permute_65);  arg101_1 = view_130 = permute_65 = None
        view_131: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [16, 512, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_49: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_45, view_131);  add_45 = view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_181: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_181, [2], correction = 0, keepdim = True)
        getitem_24: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_19: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_181, getitem_25);  convert_element_type_181 = getitem_25 = None
        add_50: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_43: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
        mul_44: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, arg102_1);  mul_43 = arg102_1 = None
        add_51: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, arg103_1);  mul_44 = arg103_1 = None
        convert_element_type_182: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_132: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_182, [8192, 1024])
        permute_66: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_36: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg105_1, view_132, permute_66);  arg105_1 = view_132 = permute_66 = None
        view_133: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [16, 512, 1024]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_134: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [16, 512, -1, 64]);  view_133 = None

        # No stacktrace found for following nodes
        permute_default_51: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_135: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_182, [8192, 1024])
        permute_68: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_37: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg107_1, view_135, permute_68);  arg107_1 = view_135 = permute_68 = None
        view_136: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [16, 512, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_137: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_136, [16, 512, -1, 64]);  view_136 = None

        # No stacktrace found for following nodes
        permute_default_52: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_138: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_182, [8192, 1024]);  convert_element_type_182 = None
        permute_70: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_38: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg109_1, view_138, permute_70);  arg109_1 = view_138 = permute_70 = None
        view_139: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [16, 512, 1024]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_140: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_139, [16, 512, -1, 64]);  view_139 = None

        # No stacktrace found for following nodes
        permute_default_53: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        _scaled_dot_product_cudnn_attention_default_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_51, permute_default_52, permute_default_53, None, False, scale = 0.125);  permute_default_51 = permute_default_52 = permute_default_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_117: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_17[0];  _scaled_dot_product_cudnn_attention_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_73: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
        clone_47: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_147: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [16, 512, 1024]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [8192, 1024]);  view_147 = None
        permute_74: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_39: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg111_1, view_148, permute_74);  arg111_1 = view_148 = permute_74 = None
        view_149: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [16, 512, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_53: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_49, view_149);  add_49 = view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_201: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_201, [2], correction = 0, keepdim = True)
        getitem_26: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_21: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_201, getitem_27);  convert_element_type_201 = getitem_27 = None
        add_54: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_45: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
        mul_46: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, arg112_1);  mul_45 = arg112_1 = None
        add_55: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, arg113_1);  mul_46 = arg113_1 = None
        convert_element_type_202: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_202, [8192, 1024]);  convert_element_type_202 = None
        permute_75: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_40: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg115_1, view_150, permute_75);  arg115_1 = view_150 = permute_75 = None
        view_151: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [16, 512, 4096]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_206: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_47: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_206, 0.5)
        mul_48: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_206, 0.7071067811865476);  convert_element_type_206 = None
        erf_6: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_56: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
        convert_element_type_207: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_207, [8192, 4096]);  convert_element_type_207 = None
        permute_76: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_41: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg117_1, view_152, permute_76);  arg117_1 = view_152 = permute_76 = None
        view_153: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [16, 512, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_57: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_53, view_153);  add_53 = view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_211: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_211, [2], correction = 0, keepdim = True)
        getitem_28: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_22: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_211, getitem_29);  convert_element_type_211 = getitem_29 = None
        add_58: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_50: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
        mul_51: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, arg118_1);  mul_50 = arg118_1 = None
        add_59: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, arg119_1);  mul_51 = arg119_1 = None
        convert_element_type_212: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_154: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_212, [8192, 1024])
        permute_77: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_42: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg121_1, view_154, permute_77);  arg121_1 = view_154 = permute_77 = None
        view_155: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [16, 512, 1024]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_156: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [16, 512, -1, 64]);  view_155 = None

        # No stacktrace found for following nodes
        permute_default_48: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_157: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_212, [8192, 1024])
        permute_79: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_43: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg123_1, view_157, permute_79);  arg123_1 = view_157 = permute_79 = None
        view_158: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [16, 512, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_159: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_158, [16, 512, -1, 64]);  view_158 = None

        # No stacktrace found for following nodes
        permute_default_49: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_160: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_212, [8192, 1024]);  convert_element_type_212 = None
        permute_81: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_44: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg125_1, view_160, permute_81);  arg125_1 = view_160 = permute_81 = None
        view_161: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [16, 512, 1024]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_162: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [16, 512, -1, 64]);  view_161 = None

        # No stacktrace found for following nodes
        permute_default_50: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        _scaled_dot_product_cudnn_attention_default_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_48, permute_default_49, permute_default_50, None, False, scale = 0.125);  permute_default_48 = permute_default_49 = permute_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_116: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_16[0];  _scaled_dot_product_cudnn_attention_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_84: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_116, [0, 2, 1, 3]);  getitem_116 = None
        clone_54: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_169: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [16, 512, 1024]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [8192, 1024]);  view_169 = None
        permute_85: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_45: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg127_1, view_170, permute_85);  arg127_1 = view_170 = permute_85 = None
        view_171: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [16, 512, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_61: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, view_171);  add_57 = view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_231: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_231, [2], correction = 0, keepdim = True)
        getitem_30: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_24: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_231, getitem_31);  convert_element_type_231 = getitem_31 = None
        add_62: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_52: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
        mul_53: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, arg128_1);  mul_52 = arg128_1 = None
        add_63: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, arg129_1);  mul_53 = arg129_1 = None
        convert_element_type_232: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_232, [8192, 1024]);  convert_element_type_232 = None
        permute_86: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_46: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg131_1, view_172, permute_86);  arg131_1 = view_172 = permute_86 = None
        view_173: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [16, 512, 4096]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_236: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_54: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, 0.5)
        mul_55: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, 0.7071067811865476);  convert_element_type_236 = None
        erf_7: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_64: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
        convert_element_type_237: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_237, [8192, 4096]);  convert_element_type_237 = None
        permute_87: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_47: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg133_1, view_174, permute_87);  arg133_1 = view_174 = permute_87 = None
        view_175: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [16, 512, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_65: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_61, view_175);  add_61 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_241: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_241, [2], correction = 0, keepdim = True)
        getitem_32: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_25: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_241, getitem_33);  convert_element_type_241 = getitem_33 = None
        add_66: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_57: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_58: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg134_1);  mul_57 = arg134_1 = None
        add_67: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg135_1);  mul_58 = arg135_1 = None
        convert_element_type_242: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_176: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_242, [8192, 1024])
        permute_88: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_48: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg137_1, view_176, permute_88);  arg137_1 = view_176 = permute_88 = None
        view_177: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [16, 512, 1024]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_178: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [16, 512, -1, 64]);  view_177 = None

        # No stacktrace found for following nodes
        permute_default_45: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_179: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_242, [8192, 1024])
        permute_90: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_49: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg139_1, view_179, permute_90);  arg139_1 = view_179 = permute_90 = None
        view_180: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [16, 512, 1024]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_181: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_180, [16, 512, -1, 64]);  view_180 = None

        # No stacktrace found for following nodes
        permute_default_46: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_182: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_242, [8192, 1024]);  convert_element_type_242 = None
        permute_92: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_50: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg141_1, view_182, permute_92);  arg141_1 = view_182 = permute_92 = None
        view_183: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [16, 512, 1024]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_184: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_183, [16, 512, -1, 64]);  view_183 = None

        # No stacktrace found for following nodes
        permute_default_47: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        _scaled_dot_product_cudnn_attention_default_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_45, permute_default_46, permute_default_47, None, False, scale = 0.125);  permute_default_45 = permute_default_46 = permute_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_115: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_15[0];  _scaled_dot_product_cudnn_attention_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_95: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        clone_61: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_191: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [16, 512, 1024]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [8192, 1024]);  view_191 = None
        permute_96: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_51: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg143_1, view_192, permute_96);  arg143_1 = view_192 = permute_96 = None
        view_193: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [16, 512, 1024]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_69: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_65, view_193);  add_65 = view_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_261: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_261, [2], correction = 0, keepdim = True)
        getitem_34: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_27: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_261, getitem_35);  convert_element_type_261 = getitem_35 = None
        add_70: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_59: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
        mul_60: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, arg144_1);  mul_59 = arg144_1 = None
        add_71: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, arg145_1);  mul_60 = arg145_1 = None
        convert_element_type_262: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_262, [8192, 1024]);  convert_element_type_262 = None
        permute_97: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_52: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg147_1, view_194, permute_97);  arg147_1 = view_194 = permute_97 = None
        view_195: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [16, 512, 4096]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_266: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_61: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_266, 0.5)
        mul_62: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_266, 0.7071067811865476);  convert_element_type_266 = None
        erf_8: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_72: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
        convert_element_type_267: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_267, [8192, 4096]);  convert_element_type_267 = None
        permute_98: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_53: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg149_1, view_196, permute_98);  arg149_1 = view_196 = permute_98 = None
        view_197: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [16, 512, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_73: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_69, view_197);  add_69 = view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_271: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_271, [2], correction = 0, keepdim = True)
        getitem_36: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_28: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_271, getitem_37);  convert_element_type_271 = getitem_37 = None
        add_74: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_64: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
        mul_65: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, arg150_1);  mul_64 = arg150_1 = None
        add_75: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, arg151_1);  mul_65 = arg151_1 = None
        convert_element_type_272: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_198: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_272, [8192, 1024])
        permute_99: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_54: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg153_1, view_198, permute_99);  arg153_1 = view_198 = permute_99 = None
        view_199: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [16, 512, 1024]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_200: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [16, 512, -1, 64]);  view_199 = None

        # No stacktrace found for following nodes
        permute_default_42: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_201: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_272, [8192, 1024])
        permute_101: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_55: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg155_1, view_201, permute_101);  arg155_1 = view_201 = permute_101 = None
        view_202: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [16, 512, 1024]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_203: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_202, [16, 512, -1, 64]);  view_202 = None

        # No stacktrace found for following nodes
        permute_default_43: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_204: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_272, [8192, 1024]);  convert_element_type_272 = None
        permute_103: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_56: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg157_1, view_204, permute_103);  arg157_1 = view_204 = permute_103 = None
        view_205: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [16, 512, 1024]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_206: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_205, [16, 512, -1, 64]);  view_205 = None

        # No stacktrace found for following nodes
        permute_default_44: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        _scaled_dot_product_cudnn_attention_default_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_42, permute_default_43, permute_default_44, None, False, scale = 0.125);  permute_default_42 = permute_default_43 = permute_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_114: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_14[0];  _scaled_dot_product_cudnn_attention_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_106: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_114, [0, 2, 1, 3]);  getitem_114 = None
        clone_68: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_213: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [16, 512, 1024]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [8192, 1024]);  view_213 = None
        permute_107: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_57: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg159_1, view_214, permute_107);  arg159_1 = view_214 = permute_107 = None
        view_215: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [16, 512, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_77: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_73, view_215);  add_73 = view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_291: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_291, [2], correction = 0, keepdim = True)
        getitem_38: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_30: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_291, getitem_39);  convert_element_type_291 = getitem_39 = None
        add_78: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_66: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
        mul_67: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg160_1);  mul_66 = arg160_1 = None
        add_79: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg161_1);  mul_67 = arg161_1 = None
        convert_element_type_292: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_292, [8192, 1024]);  convert_element_type_292 = None
        permute_108: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_58: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg163_1, view_216, permute_108);  arg163_1 = view_216 = permute_108 = None
        view_217: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [16, 512, 4096]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_296: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_68: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, 0.5)
        mul_69: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, 0.7071067811865476);  convert_element_type_296 = None
        erf_9: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_80: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
        convert_element_type_297: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_297, [8192, 4096]);  convert_element_type_297 = None
        permute_109: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_59: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg165_1, view_218, permute_109);  arg165_1 = view_218 = permute_109 = None
        view_219: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [16, 512, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_81: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_77, view_219);  add_77 = view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_301: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_301, [2], correction = 0, keepdim = True)
        getitem_40: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_31: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_301, getitem_41);  convert_element_type_301 = getitem_41 = None
        add_82: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_71: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
        mul_72: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, arg166_1);  mul_71 = arg166_1 = None
        add_83: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_72, arg167_1);  mul_72 = arg167_1 = None
        convert_element_type_302: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_220: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_302, [8192, 1024])
        permute_110: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_60: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg169_1, view_220, permute_110);  arg169_1 = view_220 = permute_110 = None
        view_221: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [16, 512, 1024]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_222: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [16, 512, -1, 64]);  view_221 = None

        # No stacktrace found for following nodes
        permute_default_39: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_223: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_302, [8192, 1024])
        permute_112: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_61: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg171_1, view_223, permute_112);  arg171_1 = view_223 = permute_112 = None
        view_224: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [16, 512, 1024]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_225: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_224, [16, 512, -1, 64]);  view_224 = None

        # No stacktrace found for following nodes
        permute_default_40: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_226: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_302, [8192, 1024]);  convert_element_type_302 = None
        permute_114: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_62: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg173_1, view_226, permute_114);  arg173_1 = view_226 = permute_114 = None
        view_227: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [16, 512, 1024]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_228: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [16, 512, -1, 64]);  view_227 = None

        # No stacktrace found for following nodes
        permute_default_41: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        _scaled_dot_product_cudnn_attention_default_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_39, permute_default_40, permute_default_41, None, False, scale = 0.125);  permute_default_39 = permute_default_40 = permute_default_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_113: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_13[0];  _scaled_dot_product_cudnn_attention_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_117: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_113, [0, 2, 1, 3]);  getitem_113 = None
        clone_75: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_235: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [16, 512, 1024]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [8192, 1024]);  view_235 = None
        permute_118: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_63: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg175_1, view_236, permute_118);  arg175_1 = view_236 = permute_118 = None
        view_237: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [16, 512, 1024]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_85: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_81, view_237);  add_81 = view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_321: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_321, [2], correction = 0, keepdim = True)
        getitem_42: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_33: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_321, getitem_43);  convert_element_type_321 = getitem_43 = None
        add_86: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_73: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
        mul_74: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg176_1);  mul_73 = arg176_1 = None
        add_87: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg177_1);  mul_74 = arg177_1 = None
        convert_element_type_322: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_322, [8192, 1024]);  convert_element_type_322 = None
        permute_119: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_64: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg179_1, view_238, permute_119);  arg179_1 = view_238 = permute_119 = None
        view_239: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [16, 512, 4096]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_326: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_75: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_326, 0.5)
        mul_76: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_326, 0.7071067811865476);  convert_element_type_326 = None
        erf_10: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_88: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
        convert_element_type_327: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_327, [8192, 4096]);  convert_element_type_327 = None
        permute_120: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_65: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg181_1, view_240, permute_120);  arg181_1 = view_240 = permute_120 = None
        view_241: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [16, 512, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_89: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_85, view_241);  add_85 = view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_331: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_331, [2], correction = 0, keepdim = True)
        getitem_44: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_34: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, getitem_45);  convert_element_type_331 = getitem_45 = None
        add_90: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_78: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_79: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, arg182_1);  mul_78 = arg182_1 = None
        add_91: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, arg183_1);  mul_79 = arg183_1 = None
        convert_element_type_332: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_242: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_332, [8192, 1024])
        permute_121: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_66: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg185_1, view_242, permute_121);  arg185_1 = view_242 = permute_121 = None
        view_243: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [16, 512, 1024]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_244: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_243, [16, 512, -1, 64]);  view_243 = None

        # No stacktrace found for following nodes
        permute_default_36: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_245: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_332, [8192, 1024])
        permute_123: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_67: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg187_1, view_245, permute_123);  arg187_1 = view_245 = permute_123 = None
        view_246: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [16, 512, 1024]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_247: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_246, [16, 512, -1, 64]);  view_246 = None

        # No stacktrace found for following nodes
        permute_default_37: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_248: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_332, [8192, 1024]);  convert_element_type_332 = None
        permute_125: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_68: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg189_1, view_248, permute_125);  arg189_1 = view_248 = permute_125 = None
        view_249: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [16, 512, 1024]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_250: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_249, [16, 512, -1, 64]);  view_249 = None

        # No stacktrace found for following nodes
        permute_default_38: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        _scaled_dot_product_cudnn_attention_default_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_36, permute_default_37, permute_default_38, None, False, scale = 0.125);  permute_default_36 = permute_default_37 = permute_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_112: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_12[0];  _scaled_dot_product_cudnn_attention_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_128: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None
        clone_82: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_257: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [16, 512, 1024]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [8192, 1024]);  view_257 = None
        permute_129: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_69: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg191_1, view_258, permute_129);  arg191_1 = view_258 = permute_129 = None
        view_259: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [16, 512, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_93: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_89, view_259);  add_89 = view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_351: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_351, [2], correction = 0, keepdim = True)
        getitem_46: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_36: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_351, getitem_47);  convert_element_type_351 = getitem_47 = None
        add_94: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_80: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
        mul_81: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, arg192_1);  mul_80 = arg192_1 = None
        add_95: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, arg193_1);  mul_81 = arg193_1 = None
        convert_element_type_352: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_352, [8192, 1024]);  convert_element_type_352 = None
        permute_130: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_70: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg195_1, view_260, permute_130);  arg195_1 = view_260 = permute_130 = None
        view_261: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [16, 512, 4096]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_356: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_82: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_356, 0.5)
        mul_83: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_356, 0.7071067811865476);  convert_element_type_356 = None
        erf_11: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_96: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
        convert_element_type_357: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_357, [8192, 4096]);  convert_element_type_357 = None
        permute_131: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_71: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg197_1, view_262, permute_131);  arg197_1 = view_262 = permute_131 = None
        view_263: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [16, 512, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_97: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_93, view_263);  add_93 = view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_361: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_361, [2], correction = 0, keepdim = True)
        getitem_48: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        sub_37: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_361, getitem_49);  convert_element_type_361 = getitem_49 = None
        add_98: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_85: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
        mul_86: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, arg198_1);  mul_85 = arg198_1 = None
        add_99: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, arg199_1);  mul_86 = arg199_1 = None
        convert_element_type_362: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_264: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_362, [8192, 1024])
        permute_132: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_72: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg201_1, view_264, permute_132);  arg201_1 = view_264 = permute_132 = None
        view_265: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [16, 512, 1024]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_266: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_265, [16, 512, -1, 64]);  view_265 = None

        # No stacktrace found for following nodes
        permute_default_33: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_267: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_362, [8192, 1024])
        permute_134: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_73: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg203_1, view_267, permute_134);  arg203_1 = view_267 = permute_134 = None
        view_268: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [16, 512, 1024]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_269: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_268, [16, 512, -1, 64]);  view_268 = None

        # No stacktrace found for following nodes
        permute_default_34: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_270: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_362, [8192, 1024]);  convert_element_type_362 = None
        permute_136: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_74: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg205_1, view_270, permute_136);  arg205_1 = view_270 = permute_136 = None
        view_271: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [16, 512, 1024]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_272: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_271, [16, 512, -1, 64]);  view_271 = None

        # No stacktrace found for following nodes
        permute_default_35: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        _scaled_dot_product_cudnn_attention_default_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_33, permute_default_34, permute_default_35, None, False, scale = 0.125);  permute_default_33 = permute_default_34 = permute_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_111: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_11[0];  _scaled_dot_product_cudnn_attention_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_139: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_111, [0, 2, 1, 3]);  getitem_111 = None
        clone_89: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_279: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [16, 512, 1024]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_279, [8192, 1024]);  view_279 = None
        permute_140: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_75: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg207_1, view_280, permute_140);  arg207_1 = view_280 = permute_140 = None
        view_281: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [16, 512, 1024]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_101: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, view_281);  add_97 = view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_381: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_381, [2], correction = 0, keepdim = True)
        getitem_50: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        sub_39: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_381, getitem_51);  convert_element_type_381 = getitem_51 = None
        add_102: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_87: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = rsqrt_25 = None
        mul_88: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, arg208_1);  mul_87 = arg208_1 = None
        add_103: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, arg209_1);  mul_88 = arg209_1 = None
        convert_element_type_382: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_282: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_382, [8192, 1024]);  convert_element_type_382 = None
        permute_141: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_76: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg211_1, view_282, permute_141);  arg211_1 = view_282 = permute_141 = None
        view_283: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [16, 512, 4096]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_386: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        mul_89: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_386, 0.5)
        mul_90: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_386, 0.7071067811865476);  convert_element_type_386 = None
        erf_12: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_104: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_91: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, add_104);  mul_89 = add_104 = None
        convert_element_type_387: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_91, torch.bfloat16);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_284: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_387, [8192, 4096]);  convert_element_type_387 = None
        permute_142: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_77: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg213_1, view_284, permute_142);  arg213_1 = view_284 = permute_142 = None
        view_285: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [16, 512, 1024]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_105: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, view_285);  add_101 = view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_391: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_391, [2], correction = 0, keepdim = True)
        getitem_52: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        sub_40: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_391, getitem_53);  convert_element_type_391 = getitem_53 = None
        add_106: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_92: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = rsqrt_26 = None
        mul_93: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, arg214_1);  mul_92 = arg214_1 = None
        add_107: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, arg215_1);  mul_93 = arg215_1 = None
        convert_element_type_392: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_286: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_392, [8192, 1024])
        permute_143: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_78: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg217_1, view_286, permute_143);  arg217_1 = view_286 = permute_143 = None
        view_287: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [16, 512, 1024]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_288: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_287, [16, 512, -1, 64]);  view_287 = None

        # No stacktrace found for following nodes
        permute_default_30: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_289: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_392, [8192, 1024])
        permute_145: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_79: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg219_1, view_289, permute_145);  arg219_1 = view_289 = permute_145 = None
        view_290: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [16, 512, 1024]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_291: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_290, [16, 512, -1, 64]);  view_290 = None

        # No stacktrace found for following nodes
        permute_default_31: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_292: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_392, [8192, 1024]);  convert_element_type_392 = None
        permute_147: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_80: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg221_1, view_292, permute_147);  arg221_1 = view_292 = permute_147 = None
        view_293: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [16, 512, 1024]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_294: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_293, [16, 512, -1, 64]);  view_293 = None

        # No stacktrace found for following nodes
        permute_default_32: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        _scaled_dot_product_cudnn_attention_default_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_30, permute_default_31, permute_default_32, None, False, scale = 0.125);  permute_default_30 = permute_default_31 = permute_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_110: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_10[0];  _scaled_dot_product_cudnn_attention_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_150: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_110, [0, 2, 1, 3]);  getitem_110 = None
        clone_96: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_301: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [16, 512, 1024]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_302: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [8192, 1024]);  view_301 = None
        permute_151: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_81: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg223_1, view_302, permute_151);  arg223_1 = view_302 = permute_151 = None
        view_303: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [16, 512, 1024]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_109: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, view_303);  add_105 = view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_411: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_411, [2], correction = 0, keepdim = True)
        getitem_54: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        sub_42: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_411, getitem_55);  convert_element_type_411 = getitem_55 = None
        add_110: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_94: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_27);  sub_42 = rsqrt_27 = None
        mul_95: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, arg224_1);  mul_94 = arg224_1 = None
        add_111: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, arg225_1);  mul_95 = arg225_1 = None
        convert_element_type_412: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_304: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_412, [8192, 1024]);  convert_element_type_412 = None
        permute_152: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_82: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg227_1, view_304, permute_152);  arg227_1 = view_304 = permute_152 = None
        view_305: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [16, 512, 4096]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_416: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        mul_96: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_416, 0.5)
        mul_97: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_416, 0.7071067811865476);  convert_element_type_416 = None
        erf_13: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_112: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_98: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, add_112);  mul_96 = add_112 = None
        convert_element_type_417: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_306: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_417, [8192, 4096]);  convert_element_type_417 = None
        permute_153: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        addmm_83: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg229_1, view_306, permute_153);  arg229_1 = view_306 = permute_153 = None
        view_307: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [16, 512, 1024]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_113: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, view_307);  add_109 = view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_421: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_421, [2], correction = 0, keepdim = True)
        getitem_56: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        sub_43: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_421, getitem_57);  convert_element_type_421 = getitem_57 = None
        add_114: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_99: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = rsqrt_28 = None
        mul_100: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, arg230_1);  mul_99 = arg230_1 = None
        add_115: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_100, arg231_1);  mul_100 = arg231_1 = None
        convert_element_type_422: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_308: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_422, [8192, 1024])
        permute_154: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_84: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg233_1, view_308, permute_154);  arg233_1 = view_308 = permute_154 = None
        view_309: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [16, 512, 1024]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_310: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [16, 512, -1, 64]);  view_309 = None

        # No stacktrace found for following nodes
        permute_default_27: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_311: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_422, [8192, 1024])
        permute_156: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_85: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg235_1, view_311, permute_156);  arg235_1 = view_311 = permute_156 = None
        view_312: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_85, [16, 512, 1024]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_313: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_312, [16, 512, -1, 64]);  view_312 = None

        # No stacktrace found for following nodes
        permute_default_28: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_314: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_422, [8192, 1024]);  convert_element_type_422 = None
        permute_158: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_86: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg237_1, view_314, permute_158);  arg237_1 = view_314 = permute_158 = None
        view_315: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [16, 512, 1024]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_316: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [16, 512, -1, 64]);  view_315 = None

        # No stacktrace found for following nodes
        permute_default_29: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        _scaled_dot_product_cudnn_attention_default_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_27, permute_default_28, permute_default_29, None, False, scale = 0.125);  permute_default_27 = permute_default_28 = permute_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_109: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_9[0];  _scaled_dot_product_cudnn_attention_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_161: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_109, [0, 2, 1, 3]);  getitem_109 = None
        clone_103: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_323: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [16, 512, 1024]);  clone_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_324: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_323, [8192, 1024]);  view_323 = None
        permute_162: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_87: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg239_1, view_324, permute_162);  arg239_1 = view_324 = permute_162 = None
        view_325: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_87, [16, 512, 1024]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_117: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_113, view_325);  add_113 = view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_441: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_441, [2], correction = 0, keepdim = True)
        getitem_58: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        sub_45: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_441, getitem_59);  convert_element_type_441 = getitem_59 = None
        add_118: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_101: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_29);  sub_45 = rsqrt_29 = None
        mul_102: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, arg240_1);  mul_101 = arg240_1 = None
        add_119: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, arg241_1);  mul_102 = arg241_1 = None
        convert_element_type_442: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_326: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_442, [8192, 1024]);  convert_element_type_442 = None
        permute_163: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_88: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg243_1, view_326, permute_163);  arg243_1 = view_326 = permute_163 = None
        view_327: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [16, 512, 4096]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_446: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        mul_103: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_446, 0.5)
        mul_104: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_446, 0.7071067811865476);  convert_element_type_446 = None
        erf_14: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_120: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_105: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, add_120);  mul_103 = add_120 = None
        convert_element_type_447: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_328: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_447, [8192, 4096]);  convert_element_type_447 = None
        permute_164: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_89: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg245_1, view_328, permute_164);  arg245_1 = view_328 = permute_164 = None
        view_329: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_89, [16, 512, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_121: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, view_329);  add_117 = view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_451: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_451, [2], correction = 0, keepdim = True)
        getitem_60: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        sub_46: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_451, getitem_61);  convert_element_type_451 = getitem_61 = None
        add_122: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_106: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = rsqrt_30 = None
        mul_107: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, arg246_1);  mul_106 = arg246_1 = None
        add_123: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_107, arg247_1);  mul_107 = arg247_1 = None
        convert_element_type_452: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_330: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_452, [8192, 1024])
        permute_165: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_90: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg249_1, view_330, permute_165);  arg249_1 = view_330 = permute_165 = None
        view_331: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [16, 512, 1024]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_332: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_331, [16, 512, -1, 64]);  view_331 = None

        # No stacktrace found for following nodes
        permute_default_24: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_333: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_452, [8192, 1024])
        permute_167: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_91: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg251_1, view_333, permute_167);  arg251_1 = view_333 = permute_167 = None
        view_334: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_91, [16, 512, 1024]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_335: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_334, [16, 512, -1, 64]);  view_334 = None

        # No stacktrace found for following nodes
        permute_default_25: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_336: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_452, [8192, 1024]);  convert_element_type_452 = None
        permute_169: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        addmm_92: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg253_1, view_336, permute_169);  arg253_1 = view_336 = permute_169 = None
        view_337: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_92, [16, 512, 1024]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_338: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_337, [16, 512, -1, 64]);  view_337 = None

        # No stacktrace found for following nodes
        permute_default_26: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        _scaled_dot_product_cudnn_attention_default_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_24, permute_default_25, permute_default_26, None, False, scale = 0.125);  permute_default_24 = permute_default_25 = permute_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_108: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_8[0];  _scaled_dot_product_cudnn_attention_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_172: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None
        clone_110: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_345: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [16, 512, 1024]);  clone_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_346: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [8192, 1024]);  view_345 = None
        permute_173: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        addmm_93: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg255_1, view_346, permute_173);  arg255_1 = view_346 = permute_173 = None
        view_347: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_93, [16, 512, 1024]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_125: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, view_347);  add_121 = view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_471: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_471, [2], correction = 0, keepdim = True)
        getitem_62: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        sub_48: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_471, getitem_63);  convert_element_type_471 = getitem_63 = None
        add_126: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_108: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_31);  sub_48 = rsqrt_31 = None
        mul_109: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, arg256_1);  mul_108 = arg256_1 = None
        add_127: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, arg257_1);  mul_109 = arg257_1 = None
        convert_element_type_472: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_348: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_472, [8192, 1024]);  convert_element_type_472 = None
        permute_174: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        addmm_94: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg259_1, view_348, permute_174);  arg259_1 = view_348 = permute_174 = None
        view_349: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [16, 512, 4096]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_476: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        mul_110: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_476, 0.5)
        mul_111: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_476, 0.7071067811865476);  convert_element_type_476 = None
        erf_15: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_128: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_112: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, add_128);  mul_110 = add_128 = None
        convert_element_type_477: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_350: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_477, [8192, 4096]);  convert_element_type_477 = None
        permute_175: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        addmm_95: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg261_1, view_350, permute_175);  arg261_1 = view_350 = permute_175 = None
        view_351: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_95, [16, 512, 1024]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_129: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, view_351);  add_125 = view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_481: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_481, [2], correction = 0, keepdim = True)
        getitem_64: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        sub_49: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_481, getitem_65);  convert_element_type_481 = getitem_65 = None
        add_130: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_113: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
        mul_114: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, arg262_1);  mul_113 = arg262_1 = None
        add_131: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, arg263_1);  mul_114 = arg263_1 = None
        convert_element_type_482: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_352: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_482, [8192, 1024])
        permute_176: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_96: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg265_1, view_352, permute_176);  arg265_1 = view_352 = permute_176 = None
        view_353: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_96, [16, 512, 1024]);  addmm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_354: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_353, [16, 512, -1, 64]);  view_353 = None

        # No stacktrace found for following nodes
        permute_default_21: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_355: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_482, [8192, 1024])
        permute_178: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_97: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg267_1, view_355, permute_178);  arg267_1 = view_355 = permute_178 = None
        view_356: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_97, [16, 512, 1024]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_357: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_356, [16, 512, -1, 64]);  view_356 = None

        # No stacktrace found for following nodes
        permute_default_22: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_358: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_482, [8192, 1024]);  convert_element_type_482 = None
        permute_180: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_98: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg269_1, view_358, permute_180);  arg269_1 = view_358 = permute_180 = None
        view_359: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_98, [16, 512, 1024]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_360: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_359, [16, 512, -1, 64]);  view_359 = None

        # No stacktrace found for following nodes
        permute_default_23: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        _scaled_dot_product_cudnn_attention_default_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_21, permute_default_22, permute_default_23, None, False, scale = 0.125);  permute_default_21 = permute_default_22 = permute_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_107: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_7[0];  _scaled_dot_product_cudnn_attention_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_183: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_107, [0, 2, 1, 3]);  getitem_107 = None
        clone_117: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_367: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [16, 512, 1024]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_368: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_367, [8192, 1024]);  view_367 = None
        permute_184: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        addmm_99: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg271_1, view_368, permute_184);  arg271_1 = view_368 = permute_184 = None
        view_369: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_99, [16, 512, 1024]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_133: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, view_369);  add_129 = view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_501: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_501, [2], correction = 0, keepdim = True)
        getitem_66: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        sub_51: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_501, getitem_67);  convert_element_type_501 = getitem_67 = None
        add_134: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_115: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = rsqrt_33 = None
        mul_116: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, arg272_1);  mul_115 = arg272_1 = None
        add_135: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, arg273_1);  mul_116 = arg273_1 = None
        convert_element_type_502: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_370: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_502, [8192, 1024]);  convert_element_type_502 = None
        permute_185: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_100: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg275_1, view_370, permute_185);  arg275_1 = view_370 = permute_185 = None
        view_371: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_100, [16, 512, 4096]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_506: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        mul_117: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_506, 0.5)
        mul_118: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_506, 0.7071067811865476);  convert_element_type_506 = None
        erf_16: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_136: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_119: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, add_136);  mul_117 = add_136 = None
        convert_element_type_507: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_372: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_507, [8192, 4096]);  convert_element_type_507 = None
        permute_186: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        addmm_101: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg277_1, view_372, permute_186);  arg277_1 = view_372 = permute_186 = None
        view_373: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_101, [16, 512, 1024]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_137: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, view_373);  add_133 = view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_511: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_511, [2], correction = 0, keepdim = True)
        getitem_68: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        sub_52: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_511, getitem_69);  convert_element_type_511 = getitem_69 = None
        add_138: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_120: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
        mul_121: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, arg278_1);  mul_120 = arg278_1 = None
        add_139: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, arg279_1);  mul_121 = arg279_1 = None
        convert_element_type_512: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_374: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_512, [8192, 1024])
        permute_187: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_102: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg281_1, view_374, permute_187);  arg281_1 = view_374 = permute_187 = None
        view_375: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_102, [16, 512, 1024]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_376: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_375, [16, 512, -1, 64]);  view_375 = None

        # No stacktrace found for following nodes
        permute_default_18: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_377: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_512, [8192, 1024])
        permute_189: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_103: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg283_1, view_377, permute_189);  arg283_1 = view_377 = permute_189 = None
        view_378: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_103, [16, 512, 1024]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_379: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_378, [16, 512, -1, 64]);  view_378 = None

        # No stacktrace found for following nodes
        permute_default_19: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_380: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_512, [8192, 1024]);  convert_element_type_512 = None
        permute_191: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_104: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg285_1, view_380, permute_191);  arg285_1 = view_380 = permute_191 = None
        view_381: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_104, [16, 512, 1024]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_382: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_381, [16, 512, -1, 64]);  view_381 = None

        # No stacktrace found for following nodes
        permute_default_20: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        _scaled_dot_product_cudnn_attention_default_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_18, permute_default_19, permute_default_20, None, False, scale = 0.125);  permute_default_18 = permute_default_19 = permute_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_106: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_6[0];  _scaled_dot_product_cudnn_attention_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_194: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None
        clone_124: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_389: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [16, 512, 1024]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_390: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_389, [8192, 1024]);  view_389 = None
        permute_195: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        addmm_105: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg287_1, view_390, permute_195);  arg287_1 = view_390 = permute_195 = None
        view_391: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_105, [16, 512, 1024]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_141: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_137, view_391);  add_137 = view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_531: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_531, [2], correction = 0, keepdim = True)
        getitem_70: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        sub_54: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_531, getitem_71);  convert_element_type_531 = getitem_71 = None
        add_142: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_122: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = rsqrt_35 = None
        mul_123: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, arg288_1);  mul_122 = arg288_1 = None
        add_143: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, arg289_1);  mul_123 = arg289_1 = None
        convert_element_type_532: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_532, [8192, 1024]);  convert_element_type_532 = None
        permute_196: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        addmm_106: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg291_1, view_392, permute_196);  arg291_1 = view_392 = permute_196 = None
        view_393: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_106, [16, 512, 4096]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_536: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_124: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_536, 0.5)
        mul_125: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_536, 0.7071067811865476);  convert_element_type_536 = None
        erf_17: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_144: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_126: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, add_144);  mul_124 = add_144 = None
        convert_element_type_537: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_394: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_537, [8192, 4096]);  convert_element_type_537 = None
        permute_197: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
        addmm_107: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg293_1, view_394, permute_197);  arg293_1 = view_394 = permute_197 = None
        view_395: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_107, [16, 512, 1024]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_145: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, view_395);  add_141 = view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_541: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_541, [2], correction = 0, keepdim = True)
        getitem_72: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        sub_55: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_541, getitem_73);  convert_element_type_541 = getitem_73 = None
        add_146: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_127: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = rsqrt_36 = None
        mul_128: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, arg294_1);  mul_127 = arg294_1 = None
        add_147: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_128, arg295_1);  mul_128 = arg295_1 = None
        convert_element_type_542: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.bfloat16);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_396: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_542, [8192, 1024])
        permute_198: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_108: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg297_1, view_396, permute_198);  arg297_1 = view_396 = permute_198 = None
        view_397: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_108, [16, 512, 1024]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_398: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [16, 512, -1, 64]);  view_397 = None

        # No stacktrace found for following nodes
        permute_default_15: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_399: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_542, [8192, 1024])
        permute_200: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_109: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg299_1, view_399, permute_200);  arg299_1 = view_399 = permute_200 = None
        view_400: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_109, [16, 512, 1024]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_401: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_400, [16, 512, -1, 64]);  view_400 = None

        # No stacktrace found for following nodes
        permute_default_16: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_402: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_542, [8192, 1024]);  convert_element_type_542 = None
        permute_202: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_110: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg301_1, view_402, permute_202);  arg301_1 = view_402 = permute_202 = None
        view_403: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_110, [16, 512, 1024]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_404: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_403, [16, 512, -1, 64]);  view_403 = None

        # No stacktrace found for following nodes
        permute_default_17: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        _scaled_dot_product_cudnn_attention_default_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_15, permute_default_16, permute_default_17, None, False, scale = 0.125);  permute_default_15 = permute_default_16 = permute_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_105: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_5[0];  _scaled_dot_product_cudnn_attention_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_205: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_105, [0, 2, 1, 3]);  getitem_105 = None
        clone_131: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_411: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [16, 512, 1024]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_412: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [8192, 1024]);  view_411 = None
        permute_206: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        addmm_111: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg303_1, view_412, permute_206);  arg303_1 = view_412 = permute_206 = None
        view_413: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_111, [16, 512, 1024]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_149: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, view_413);  add_145 = view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_561: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_561, [2], correction = 0, keepdim = True)
        getitem_74: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        sub_57: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_561, getitem_75);  convert_element_type_561 = getitem_75 = None
        add_150: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        mul_129: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = rsqrt_37 = None
        mul_130: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, arg304_1);  mul_129 = arg304_1 = None
        add_151: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, arg305_1);  mul_130 = arg305_1 = None
        convert_element_type_562: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.bfloat16);  add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_414: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_562, [8192, 1024]);  convert_element_type_562 = None
        permute_207: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        addmm_112: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg307_1, view_414, permute_207);  arg307_1 = view_414 = permute_207 = None
        view_415: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_112, [16, 512, 4096]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_566: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_131: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, 0.5)
        mul_132: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, 0.7071067811865476);  convert_element_type_566 = None
        erf_18: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_152: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_133: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, add_152);  mul_131 = add_152 = None
        convert_element_type_567: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_416: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_567, [8192, 4096]);  convert_element_type_567 = None
        permute_208: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        addmm_113: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg309_1, view_416, permute_208);  arg309_1 = view_416 = permute_208 = None
        view_417: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_113, [16, 512, 1024]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_153: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_149, view_417);  add_149 = view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_571: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_153, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_571, [2], correction = 0, keepdim = True)
        getitem_76: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        sub_58: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_571, getitem_77);  convert_element_type_571 = getitem_77 = None
        add_154: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        mul_134: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = rsqrt_38 = None
        mul_135: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, arg310_1);  mul_134 = arg310_1 = None
        add_155: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_135, arg311_1);  mul_135 = arg311_1 = None
        convert_element_type_572: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16);  add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_418: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_572, [8192, 1024])
        permute_209: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        addmm_114: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg313_1, view_418, permute_209);  arg313_1 = view_418 = permute_209 = None
        view_419: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_114, [16, 512, 1024]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_420: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_419, [16, 512, -1, 64]);  view_419 = None

        # No stacktrace found for following nodes
        permute_default_12: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_421: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_572, [8192, 1024])
        permute_211: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
        addmm_115: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg315_1, view_421, permute_211);  arg315_1 = view_421 = permute_211 = None
        view_422: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_115, [16, 512, 1024]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_423: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_422, [16, 512, -1, 64]);  view_422 = None

        # No stacktrace found for following nodes
        permute_default_13: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_424: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_572, [8192, 1024]);  convert_element_type_572 = None
        permute_213: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
        addmm_116: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg317_1, view_424, permute_213);  arg317_1 = view_424 = permute_213 = None
        view_425: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_116, [16, 512, 1024]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_426: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_425, [16, 512, -1, 64]);  view_425 = None

        # No stacktrace found for following nodes
        permute_default_14: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
        _scaled_dot_product_cudnn_attention_default_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_12, permute_default_13, permute_default_14, None, False, scale = 0.125);  permute_default_12 = permute_default_13 = permute_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_104: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_4[0];  _scaled_dot_product_cudnn_attention_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_216: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        clone_138: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_433: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [16, 512, 1024]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_434: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_433, [8192, 1024]);  view_433 = None
        permute_217: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        addmm_117: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg319_1, view_434, permute_217);  arg319_1 = view_434 = permute_217 = None
        view_435: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_117, [16, 512, 1024]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_157: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_153, view_435);  add_153 = view_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_591: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_591, [2], correction = 0, keepdim = True)
        getitem_78: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        sub_60: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_591, getitem_79);  convert_element_type_591 = getitem_79 = None
        add_158: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        mul_136: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_39);  sub_60 = rsqrt_39 = None
        mul_137: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, arg320_1);  mul_136 = arg320_1 = None
        add_159: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, arg321_1);  mul_137 = arg321_1 = None
        convert_element_type_592: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_436: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_592, [8192, 1024]);  convert_element_type_592 = None
        permute_218: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        addmm_118: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg323_1, view_436, permute_218);  arg323_1 = view_436 = permute_218 = None
        view_437: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_118, [16, 512, 4096]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_596: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        mul_138: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_596, 0.5)
        mul_139: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_596, 0.7071067811865476);  convert_element_type_596 = None
        erf_19: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
        add_160: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_140: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, add_160);  mul_138 = add_160 = None
        convert_element_type_597: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_140, torch.bfloat16);  mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_438: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_597, [8192, 4096]);  convert_element_type_597 = None
        permute_219: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        addmm_119: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg325_1, view_438, permute_219);  arg325_1 = view_438 = permute_219 = None
        view_439: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_119, [16, 512, 1024]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_161: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_157, view_439);  add_157 = view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_601: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_601, [2], correction = 0, keepdim = True)
        getitem_80: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        sub_61: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_601, getitem_81);  convert_element_type_601 = getitem_81 = None
        add_162: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_141: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = rsqrt_40 = None
        mul_142: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, arg326_1);  mul_141 = arg326_1 = None
        add_163: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, arg327_1);  mul_142 = arg327_1 = None
        convert_element_type_602: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_163, torch.bfloat16);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_440: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_602, [8192, 1024])
        permute_220: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_120: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg329_1, view_440, permute_220);  arg329_1 = view_440 = permute_220 = None
        view_441: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_120, [16, 512, 1024]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_442: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [16, 512, -1, 64]);  view_441 = None

        # No stacktrace found for following nodes
        permute_default_9: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_443: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_602, [8192, 1024])
        permute_222: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        addmm_121: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg331_1, view_443, permute_222);  arg331_1 = view_443 = permute_222 = None
        view_444: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_121, [16, 512, 1024]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_445: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_444, [16, 512, -1, 64]);  view_444 = None

        # No stacktrace found for following nodes
        permute_default_10: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_446: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_602, [8192, 1024]);  convert_element_type_602 = None
        permute_224: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        addmm_122: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg333_1, view_446, permute_224);  arg333_1 = view_446 = permute_224 = None
        view_447: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_122, [16, 512, 1024]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_448: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_447, [16, 512, -1, 64]);  view_447 = None

        # No stacktrace found for following nodes
        permute_default_11: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        _scaled_dot_product_cudnn_attention_default_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_9, permute_default_10, permute_default_11, None, False, scale = 0.125);  permute_default_9 = permute_default_10 = permute_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_103: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_3[0];  _scaled_dot_product_cudnn_attention_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_227: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_103, [0, 2, 1, 3]);  getitem_103 = None
        clone_145: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_455: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [16, 512, 1024]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_456: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_455, [8192, 1024]);  view_455 = None
        permute_228: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
        addmm_123: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg335_1, view_456, permute_228);  arg335_1 = view_456 = permute_228 = None
        view_457: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_123, [16, 512, 1024]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_165: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_161, view_457);  add_161 = view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_621: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_165, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_621, [2], correction = 0, keepdim = True)
        getitem_82: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        sub_63: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_621, getitem_83);  convert_element_type_621 = getitem_83 = None
        add_166: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_143: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_41);  sub_63 = rsqrt_41 = None
        mul_144: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, arg336_1);  mul_143 = arg336_1 = None
        add_167: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, arg337_1);  mul_144 = arg337_1 = None
        convert_element_type_622: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.bfloat16);  add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_458: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_622, [8192, 1024]);  convert_element_type_622 = None
        permute_229: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        addmm_124: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg339_1, view_458, permute_229);  arg339_1 = view_458 = permute_229 = None
        view_459: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_124, [16, 512, 4096]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_626: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        mul_145: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, 0.5)
        mul_146: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, 0.7071067811865476);  convert_element_type_626 = None
        erf_20: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
        add_168: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_147: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, add_168);  mul_145 = add_168 = None
        convert_element_type_627: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_460: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_627, [8192, 4096]);  convert_element_type_627 = None
        permute_230: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
        addmm_125: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg341_1, view_460, permute_230);  arg341_1 = view_460 = permute_230 = None
        view_461: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_125, [16, 512, 1024]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_169: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_165, view_461);  add_165 = view_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_631: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_631, [2], correction = 0, keepdim = True)
        getitem_84: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        sub_64: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_631, getitem_85);  convert_element_type_631 = getitem_85 = None
        add_170: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_148: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = rsqrt_42 = None
        mul_149: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, arg342_1);  mul_148 = arg342_1 = None
        add_171: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, arg343_1);  mul_149 = arg343_1 = None
        convert_element_type_632: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16);  add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_462: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_632, [8192, 1024])
        permute_231: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        addmm_126: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg345_1, view_462, permute_231);  arg345_1 = view_462 = permute_231 = None
        view_463: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_126, [16, 512, 1024]);  addmm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_464: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_463, [16, 512, -1, 64]);  view_463 = None

        # No stacktrace found for following nodes
        permute_default_6: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_465: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_632, [8192, 1024])
        permute_233: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
        addmm_127: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg347_1, view_465, permute_233);  arg347_1 = view_465 = permute_233 = None
        view_466: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_127, [16, 512, 1024]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_467: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_466, [16, 512, -1, 64]);  view_466 = None

        # No stacktrace found for following nodes
        permute_default_7: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_468: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_632, [8192, 1024]);  convert_element_type_632 = None
        permute_235: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg348_1, [1, 0]);  arg348_1 = None
        addmm_128: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg349_1, view_468, permute_235);  arg349_1 = view_468 = permute_235 = None
        view_469: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_128, [16, 512, 1024]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_470: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_469, [16, 512, -1, 64]);  view_469 = None

        # No stacktrace found for following nodes
        permute_default_8: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        _scaled_dot_product_cudnn_attention_default_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_6, permute_default_7, permute_default_8, None, False, scale = 0.125);  permute_default_6 = permute_default_7 = permute_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_102: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_2[0];  _scaled_dot_product_cudnn_attention_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_238: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_102, [0, 2, 1, 3]);  getitem_102 = None
        clone_152: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_477: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_152, [16, 512, 1024]);  clone_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_478: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_477, [8192, 1024]);  view_477 = None
        permute_239: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
        addmm_129: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg351_1, view_478, permute_239);  arg351_1 = view_478 = permute_239 = None
        view_479: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_129, [16, 512, 1024]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_173: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_169, view_479);  add_169 = view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_651: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_651, [2], correction = 0, keepdim = True)
        getitem_86: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        sub_66: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_651, getitem_87);  convert_element_type_651 = getitem_87 = None
        add_174: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_150: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_43);  sub_66 = rsqrt_43 = None
        mul_151: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, arg352_1);  mul_150 = arg352_1 = None
        add_175: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, arg353_1);  mul_151 = arg353_1 = None
        convert_element_type_652: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_480: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_652, [8192, 1024]);  convert_element_type_652 = None
        permute_240: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
        addmm_130: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg355_1, view_480, permute_240);  arg355_1 = view_480 = permute_240 = None
        view_481: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_130, [16, 512, 4096]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_656: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_481, torch.float32);  view_481 = None
        mul_152: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_656, 0.5)
        mul_153: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_656, 0.7071067811865476);  convert_element_type_656 = None
        erf_21: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_176: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_154: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, add_176);  mul_152 = add_176 = None
        convert_element_type_657: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_482: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_657, [8192, 4096]);  convert_element_type_657 = None
        permute_241: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
        addmm_131: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg357_1, view_482, permute_241);  arg357_1 = view_482 = permute_241 = None
        view_483: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_131, [16, 512, 1024]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_177: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_173, view_483);  add_173 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_661: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_661, [2], correction = 0, keepdim = True)
        getitem_88: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        sub_67: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_661, getitem_89);  convert_element_type_661 = getitem_89 = None
        add_178: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_155: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = rsqrt_44 = None
        mul_156: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, arg358_1);  mul_155 = arg358_1 = None
        add_179: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_156, arg359_1);  mul_156 = arg359_1 = None
        convert_element_type_662: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_484: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_662, [8192, 1024])
        permute_242: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
        addmm_132: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg361_1, view_484, permute_242);  arg361_1 = view_484 = permute_242 = None
        view_485: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_132, [16, 512, 1024]);  addmm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_486: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [16, 512, -1, 64]);  view_485 = None

        # No stacktrace found for following nodes
        permute_default_3: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_487: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_662, [8192, 1024])
        permute_244: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
        addmm_133: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg363_1, view_487, permute_244);  arg363_1 = view_487 = permute_244 = None
        view_488: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_133, [16, 512, 1024]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_489: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_488, [16, 512, -1, 64]);  view_488 = None

        # No stacktrace found for following nodes
        permute_default_4: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_490: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_662, [8192, 1024]);  convert_element_type_662 = None
        permute_246: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
        addmm_134: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg365_1, view_490, permute_246);  arg365_1 = view_490 = permute_246 = None
        view_491: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_134, [16, 512, 1024]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_492: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_491, [16, 512, -1, 64]);  view_491 = None

        # No stacktrace found for following nodes
        permute_default_5: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
        _scaled_dot_product_cudnn_attention_default_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_3, permute_default_4, permute_default_5, None, False, scale = 0.125);  permute_default_3 = permute_default_4 = permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_101: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_1[0];  _scaled_dot_product_cudnn_attention_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_249: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
        clone_159: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_499: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_159, [16, 512, 1024]);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_500: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_499, [8192, 1024]);  view_499 = None
        permute_250: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg366_1, [1, 0]);  arg366_1 = None
        addmm_135: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg367_1, view_500, permute_250);  arg367_1 = view_500 = permute_250 = None
        view_501: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_135, [16, 512, 1024]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_181: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_177, view_501);  add_177 = view_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_681: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_181, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_681, [2], correction = 0, keepdim = True)
        getitem_90: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        sub_69: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_681, getitem_91);  convert_element_type_681 = getitem_91 = None
        add_182: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        mul_157: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_45);  sub_69 = rsqrt_45 = None
        mul_158: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, arg368_1);  mul_157 = arg368_1 = None
        add_183: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, arg369_1);  mul_158 = arg369_1 = None
        convert_element_type_682: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_183, torch.bfloat16);  add_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_502: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_682, [8192, 1024]);  convert_element_type_682 = None
        permute_251: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
        addmm_136: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg371_1, view_502, permute_251);  arg371_1 = view_502 = permute_251 = None
        view_503: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_136, [16, 512, 4096]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_686: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        mul_159: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_686, 0.5)
        mul_160: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_686, 0.7071067811865476);  convert_element_type_686 = None
        erf_22: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_184: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_161: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_159, add_184);  mul_159 = add_184 = None
        convert_element_type_687: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_687, [8192, 4096]);  convert_element_type_687 = None
        permute_252: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg372_1, [1, 0]);  arg372_1 = None
        addmm_137: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg373_1, view_504, permute_252);  arg373_1 = view_504 = permute_252 = None
        view_505: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_137, [16, 512, 1024]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_185: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_181, view_505);  add_181 = view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        convert_element_type_691: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_185, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_691, [2], correction = 0, keepdim = True)
        getitem_92: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_46[0]
        getitem_93: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        sub_70: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_691, getitem_93);  convert_element_type_691 = getitem_93 = None
        add_186: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        mul_162: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = rsqrt_46 = None
        mul_163: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, arg374_1);  mul_162 = arg374_1 = None
        add_187: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_163, arg375_1);  mul_163 = arg375_1 = None
        convert_element_type_692: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_187, torch.bfloat16);  add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_506: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_692, [8192, 1024])
        permute_253: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
        addmm_138: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg377_1, view_506, permute_253);  arg377_1 = view_506 = permute_253 = None
        view_507: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_138, [16, 512, 1024]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_508: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_507, [16, 512, -1, 64]);  view_507 = None

        # No stacktrace found for following nodes
        permute_default: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        view_509: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_692, [8192, 1024])
        permute_255: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        addmm_139: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg379_1, view_509, permute_255);  arg379_1 = view_509 = permute_255 = None
        view_510: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_139, [16, 512, 1024]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_511: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_510, [16, 512, -1, 64]);  view_510 = None

        # No stacktrace found for following nodes
        permute_default_1: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_512: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_692, [8192, 1024]);  convert_element_type_692 = None
        permute_257: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        addmm_140: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg381_1, view_512, permute_257);  arg381_1 = view_512 = permute_257 = None
        view_513: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_140, [16, 512, 1024]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_514: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_513, [16, 512, -1, 64]);  view_513 = None

        # No stacktrace found for following nodes
        permute_default_2: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        _scaled_dot_product_cudnn_attention_default = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default, permute_default_1, permute_default_2, None, False, scale = 0.125);  permute_default = permute_default_1 = permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_100: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default[0];  _scaled_dot_product_cudnn_attention_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_260: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None
        clone_166: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_521: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_166, [16, 512, 1024]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_522: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_521, [8192, 1024]);  view_521 = None
        permute_261: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg382_1, [1, 0]);  arg382_1 = None
        addmm_141: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg383_1, view_522, permute_261);  arg383_1 = view_522 = permute_261 = None
        view_523: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_141, [16, 512, 1024]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_189: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_185, view_523);  add_185 = view_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        convert_element_type_711: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_711, [2], correction = 0, keepdim = True)
        getitem_94: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_47[0]
        getitem_95: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        sub_72: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_711, getitem_95);  convert_element_type_711 = getitem_95 = None
        add_190: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        mul_164: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_47);  sub_72 = rsqrt_47 = None
        mul_165: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, arg384_1);  mul_164 = arg384_1 = None
        add_191: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, arg385_1);  mul_165 = arg385_1 = None
        convert_element_type_712: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16);  add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_524: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_712, [8192, 1024]);  convert_element_type_712 = None
        permute_262: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
        addmm_142: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg387_1, view_524, permute_262);  arg387_1 = view_524 = permute_262 = None
        view_525: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_142, [16, 512, 4096]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_716: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_525, torch.float32);  view_525 = None
        mul_166: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_716, 0.5)
        mul_167: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_716, 0.7071067811865476);  convert_element_type_716 = None
        erf_23: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_192: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_168: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, add_192);  mul_166 = add_192 = None
        convert_element_type_717: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_526: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_717, [8192, 4096]);  convert_element_type_717 = None
        permute_263: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg388_1, [1, 0]);  arg388_1 = None
        addmm_143: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg389_1, view_526, permute_263);  arg389_1 = view_526 = permute_263 = None
        view_527: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_143, [16, 512, 1024]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_193: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_189, view_527);  add_189 = view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        convert_element_type_721: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_193, torch.float32);  add_193 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_721, [2], correction = 0, keepdim = True)
        getitem_96: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_48[0]
        getitem_97: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        sub_73: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_721, getitem_97);  convert_element_type_721 = getitem_97 = None
        add_194: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_48: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        mul_169: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_48);  sub_73 = rsqrt_48 = None
        mul_170: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, arg390_1);  mul_169 = arg390_1 = None
        add_195: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, arg391_1);  mul_170 = arg391_1 = None
        convert_element_type_722: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_195, torch.bfloat16);  add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        view_528: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_722, [8192, 1024]);  convert_element_type_722 = None
        permute_264: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg392_1, [1, 0]);  arg392_1 = None
        addmm_144: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg393_1, view_528, permute_264);  arg393_1 = view_528 = permute_264 = None
        view_529: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_144, [16, 512, 1024]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_726: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        mul_171: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_726, 0.5)
        mul_172: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_726, 0.7071067811865476);  convert_element_type_726 = None
        erf_24: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
        add_196: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_173: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, add_196);  mul_171 = add_196 = None
        convert_element_type_727: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_728: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_727, torch.float32);  convert_element_type_727 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_728, [2], correction = 0, keepdim = True)
        getitem_98: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_49[0]
        getitem_99: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[16, 513][513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1], -100.0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[16, 512][513, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_169: "i64[16, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_533: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [-1]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_1: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_533, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_74: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_728, getitem_99);  convert_element_type_728 = getitem_99 = None
        add_197: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
        rsqrt_49: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        mul_174: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = rsqrt_49 = None
        mul_175: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, arg394_1);  mul_174 = arg394_1 = None
        add_198: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_175, arg395_1);  mul_175 = arg395_1 = None
        convert_element_type_729: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16);  add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        view_530: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_729, [8192, 1024]);  convert_element_type_729 = None
        permute_265: "bf16[1024, 29056][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm_145: "bf16[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.addmm.default(arg396_1, view_530, permute_265);  arg396_1 = view_530 = permute_265 = None
        view_531: "bf16[16, 512, 29056][14876672, 29056, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_145, [16, 512, 29056]);  addmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_733: "f32[16, 512, 29056][14876672, 29056, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_531, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_532: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_733, [-1, 29056]);  convert_element_type_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_24: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(view_532, [1], True)
        sub_75: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = amax_24 = None
        exp_24: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.exp.default(sub_75)
        sum_25: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_76: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = log = None
        ne: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_533, -100)
        full_default_2: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne, view_533, full_default_2);  ne = full_default_2 = None
        unsqueeze_2: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(sub_76, 1, unsqueeze_2);  sub_76 = unsqueeze_2 = None
        squeeze: "f32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        sum_27: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_533, -100);  view_533 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_734: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        div_48: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_734);  sum_27 = convert_element_type_734 = None
        return (div_48, view_531)

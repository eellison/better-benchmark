class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 128][128, 1]cuda:0", arg1_1: "bf16[256008, 1024][1024, 1]cuda:0", arg2_1: "bf16[2050, 1024][1024, 1]cuda:0", arg3_1: "bf16[1024][1]cuda:0", arg4_1: "bf16[1024][1]cuda:0", arg5_1: "bf16[1024, 1024][1024, 1]cuda:0", arg6_1: "bf16[1024][1]cuda:0", arg7_1: "bf16[1024, 1024][1024, 1]cuda:0", arg8_1: "bf16[1024][1]cuda:0", arg9_1: "bf16[1024, 1024][1024, 1]cuda:0", arg10_1: "bf16[1024][1]cuda:0", arg11_1: "bf16[1024, 1024][1024, 1]cuda:0", arg12_1: "bf16[1024][1]cuda:0", arg13_1: "bf16[1024][1]cuda:0", arg14_1: "bf16[1024][1]cuda:0", arg15_1: "bf16[4096, 1024][1024, 1]cuda:0", arg16_1: "bf16[4096][1]cuda:0", arg17_1: "bf16[1024, 4096][4096, 1]cuda:0", arg18_1: "bf16[1024][1]cuda:0", arg19_1: "bf16[1024][1]cuda:0", arg20_1: "bf16[1024][1]cuda:0", arg21_1: "bf16[1024, 1024][1024, 1]cuda:0", arg22_1: "bf16[1024][1]cuda:0", arg23_1: "bf16[1024, 1024][1024, 1]cuda:0", arg24_1: "bf16[1024][1]cuda:0", arg25_1: "bf16[1024, 1024][1024, 1]cuda:0", arg26_1: "bf16[1024][1]cuda:0", arg27_1: "bf16[1024, 1024][1024, 1]cuda:0", arg28_1: "bf16[1024][1]cuda:0", arg29_1: "bf16[1024][1]cuda:0", arg30_1: "bf16[1024][1]cuda:0", arg31_1: "bf16[4096, 1024][1024, 1]cuda:0", arg32_1: "bf16[4096][1]cuda:0", arg33_1: "bf16[1024, 4096][4096, 1]cuda:0", arg34_1: "bf16[1024][1]cuda:0", arg35_1: "bf16[1024][1]cuda:0", arg36_1: "bf16[1024][1]cuda:0", arg37_1: "bf16[1024, 1024][1024, 1]cuda:0", arg38_1: "bf16[1024][1]cuda:0", arg39_1: "bf16[1024, 1024][1024, 1]cuda:0", arg40_1: "bf16[1024][1]cuda:0", arg41_1: "bf16[1024, 1024][1024, 1]cuda:0", arg42_1: "bf16[1024][1]cuda:0", arg43_1: "bf16[1024, 1024][1024, 1]cuda:0", arg44_1: "bf16[1024][1]cuda:0", arg45_1: "bf16[1024][1]cuda:0", arg46_1: "bf16[1024][1]cuda:0", arg47_1: "bf16[4096, 1024][1024, 1]cuda:0", arg48_1: "bf16[4096][1]cuda:0", arg49_1: "bf16[1024, 4096][4096, 1]cuda:0", arg50_1: "bf16[1024][1]cuda:0", arg51_1: "bf16[1024][1]cuda:0", arg52_1: "bf16[1024][1]cuda:0", arg53_1: "bf16[1024, 1024][1024, 1]cuda:0", arg54_1: "bf16[1024][1]cuda:0", arg55_1: "bf16[1024, 1024][1024, 1]cuda:0", arg56_1: "bf16[1024][1]cuda:0", arg57_1: "bf16[1024, 1024][1024, 1]cuda:0", arg58_1: "bf16[1024][1]cuda:0", arg59_1: "bf16[1024, 1024][1024, 1]cuda:0", arg60_1: "bf16[1024][1]cuda:0", arg61_1: "bf16[1024][1]cuda:0", arg62_1: "bf16[1024][1]cuda:0", arg63_1: "bf16[4096, 1024][1024, 1]cuda:0", arg64_1: "bf16[4096][1]cuda:0", arg65_1: "bf16[1024, 4096][4096, 1]cuda:0", arg66_1: "bf16[1024][1]cuda:0", arg67_1: "bf16[1024][1]cuda:0", arg68_1: "bf16[1024][1]cuda:0", arg69_1: "bf16[1024, 1024][1024, 1]cuda:0", arg70_1: "bf16[1024][1]cuda:0", arg71_1: "bf16[1024, 1024][1024, 1]cuda:0", arg72_1: "bf16[1024][1]cuda:0", arg73_1: "bf16[1024, 1024][1024, 1]cuda:0", arg74_1: "bf16[1024][1]cuda:0", arg75_1: "bf16[1024, 1024][1024, 1]cuda:0", arg76_1: "bf16[1024][1]cuda:0", arg77_1: "bf16[1024][1]cuda:0", arg78_1: "bf16[1024][1]cuda:0", arg79_1: "bf16[4096, 1024][1024, 1]cuda:0", arg80_1: "bf16[4096][1]cuda:0", arg81_1: "bf16[1024, 4096][4096, 1]cuda:0", arg82_1: "bf16[1024][1]cuda:0", arg83_1: "bf16[1024][1]cuda:0", arg84_1: "bf16[1024][1]cuda:0", arg85_1: "bf16[1024, 1024][1024, 1]cuda:0", arg86_1: "bf16[1024][1]cuda:0", arg87_1: "bf16[1024, 1024][1024, 1]cuda:0", arg88_1: "bf16[1024][1]cuda:0", arg89_1: "bf16[1024, 1024][1024, 1]cuda:0", arg90_1: "bf16[1024][1]cuda:0", arg91_1: "bf16[1024, 1024][1024, 1]cuda:0", arg92_1: "bf16[1024][1]cuda:0", arg93_1: "bf16[1024][1]cuda:0", arg94_1: "bf16[1024][1]cuda:0", arg95_1: "bf16[4096, 1024][1024, 1]cuda:0", arg96_1: "bf16[4096][1]cuda:0", arg97_1: "bf16[1024, 4096][4096, 1]cuda:0", arg98_1: "bf16[1024][1]cuda:0", arg99_1: "bf16[1024][1]cuda:0", arg100_1: "bf16[1024][1]cuda:0", arg101_1: "bf16[1024, 1024][1024, 1]cuda:0", arg102_1: "bf16[1024][1]cuda:0", arg103_1: "bf16[1024, 1024][1024, 1]cuda:0", arg104_1: "bf16[1024][1]cuda:0", arg105_1: "bf16[1024, 1024][1024, 1]cuda:0", arg106_1: "bf16[1024][1]cuda:0", arg107_1: "bf16[1024, 1024][1024, 1]cuda:0", arg108_1: "bf16[1024][1]cuda:0", arg109_1: "bf16[1024][1]cuda:0", arg110_1: "bf16[1024][1]cuda:0", arg111_1: "bf16[4096, 1024][1024, 1]cuda:0", arg112_1: "bf16[4096][1]cuda:0", arg113_1: "bf16[1024, 4096][4096, 1]cuda:0", arg114_1: "bf16[1024][1]cuda:0", arg115_1: "bf16[1024][1]cuda:0", arg116_1: "bf16[1024][1]cuda:0", arg117_1: "bf16[1024, 1024][1024, 1]cuda:0", arg118_1: "bf16[1024][1]cuda:0", arg119_1: "bf16[1024, 1024][1024, 1]cuda:0", arg120_1: "bf16[1024][1]cuda:0", arg121_1: "bf16[1024, 1024][1024, 1]cuda:0", arg122_1: "bf16[1024][1]cuda:0", arg123_1: "bf16[1024, 1024][1024, 1]cuda:0", arg124_1: "bf16[1024][1]cuda:0", arg125_1: "bf16[1024][1]cuda:0", arg126_1: "bf16[1024][1]cuda:0", arg127_1: "bf16[4096, 1024][1024, 1]cuda:0", arg128_1: "bf16[4096][1]cuda:0", arg129_1: "bf16[1024, 4096][4096, 1]cuda:0", arg130_1: "bf16[1024][1]cuda:0", arg131_1: "bf16[1024][1]cuda:0", arg132_1: "bf16[1024][1]cuda:0", arg133_1: "bf16[1024, 1024][1024, 1]cuda:0", arg134_1: "bf16[1024][1]cuda:0", arg135_1: "bf16[1024, 1024][1024, 1]cuda:0", arg136_1: "bf16[1024][1]cuda:0", arg137_1: "bf16[1024, 1024][1024, 1]cuda:0", arg138_1: "bf16[1024][1]cuda:0", arg139_1: "bf16[1024, 1024][1024, 1]cuda:0", arg140_1: "bf16[1024][1]cuda:0", arg141_1: "bf16[1024][1]cuda:0", arg142_1: "bf16[1024][1]cuda:0", arg143_1: "bf16[4096, 1024][1024, 1]cuda:0", arg144_1: "bf16[4096][1]cuda:0", arg145_1: "bf16[1024, 4096][4096, 1]cuda:0", arg146_1: "bf16[1024][1]cuda:0", arg147_1: "bf16[1024][1]cuda:0", arg148_1: "bf16[1024][1]cuda:0", arg149_1: "bf16[1024, 1024][1024, 1]cuda:0", arg150_1: "bf16[1024][1]cuda:0", arg151_1: "bf16[1024, 1024][1024, 1]cuda:0", arg152_1: "bf16[1024][1]cuda:0", arg153_1: "bf16[1024, 1024][1024, 1]cuda:0", arg154_1: "bf16[1024][1]cuda:0", arg155_1: "bf16[1024, 1024][1024, 1]cuda:0", arg156_1: "bf16[1024][1]cuda:0", arg157_1: "bf16[1024][1]cuda:0", arg158_1: "bf16[1024][1]cuda:0", arg159_1: "bf16[4096, 1024][1024, 1]cuda:0", arg160_1: "bf16[4096][1]cuda:0", arg161_1: "bf16[1024, 4096][4096, 1]cuda:0", arg162_1: "bf16[1024][1]cuda:0", arg163_1: "bf16[1024][1]cuda:0", arg164_1: "bf16[1024][1]cuda:0", arg165_1: "bf16[1024, 1024][1024, 1]cuda:0", arg166_1: "bf16[1024][1]cuda:0", arg167_1: "bf16[1024, 1024][1024, 1]cuda:0", arg168_1: "bf16[1024][1]cuda:0", arg169_1: "bf16[1024, 1024][1024, 1]cuda:0", arg170_1: "bf16[1024][1]cuda:0", arg171_1: "bf16[1024, 1024][1024, 1]cuda:0", arg172_1: "bf16[1024][1]cuda:0", arg173_1: "bf16[1024][1]cuda:0", arg174_1: "bf16[1024][1]cuda:0", arg175_1: "bf16[4096, 1024][1024, 1]cuda:0", arg176_1: "bf16[4096][1]cuda:0", arg177_1: "bf16[1024, 4096][4096, 1]cuda:0", arg178_1: "bf16[1024][1]cuda:0", arg179_1: "bf16[1024][1]cuda:0", arg180_1: "bf16[1024][1]cuda:0", arg181_1: "bf16[1024, 1024][1024, 1]cuda:0", arg182_1: "bf16[1024][1]cuda:0", arg183_1: "bf16[1024, 1024][1024, 1]cuda:0", arg184_1: "bf16[1024][1]cuda:0", arg185_1: "bf16[1024, 1024][1024, 1]cuda:0", arg186_1: "bf16[1024][1]cuda:0", arg187_1: "bf16[1024, 1024][1024, 1]cuda:0", arg188_1: "bf16[1024][1]cuda:0", arg189_1: "bf16[1024][1]cuda:0", arg190_1: "bf16[1024][1]cuda:0", arg191_1: "bf16[4096, 1024][1024, 1]cuda:0", arg192_1: "bf16[4096][1]cuda:0", arg193_1: "bf16[1024, 4096][4096, 1]cuda:0", arg194_1: "bf16[1024][1]cuda:0", arg195_1: "bf16[1024][1]cuda:0", arg196_1: "bf16[1024][1]cuda:0", arg197_1: "bf16[1024, 1024][1024, 1]cuda:0", arg198_1: "bf16[1024][1]cuda:0", arg199_1: "bf16[1024, 1024][1024, 1]cuda:0", arg200_1: "bf16[1024][1]cuda:0", arg201_1: "bf16[1024, 1024][1024, 1]cuda:0", arg202_1: "bf16[1024][1]cuda:0", arg203_1: "bf16[1024, 1024][1024, 1]cuda:0", arg204_1: "bf16[1024][1]cuda:0", arg205_1: "bf16[1024][1]cuda:0", arg206_1: "bf16[1024][1]cuda:0", arg207_1: "bf16[4096, 1024][1024, 1]cuda:0", arg208_1: "bf16[4096][1]cuda:0", arg209_1: "bf16[1024, 4096][4096, 1]cuda:0", arg210_1: "bf16[1024][1]cuda:0", arg211_1: "bf16[1024][1]cuda:0", arg212_1: "bf16[1024][1]cuda:0", arg213_1: "bf16[1024, 1024][1024, 1]cuda:0", arg214_1: "bf16[1024][1]cuda:0", arg215_1: "bf16[1024, 1024][1024, 1]cuda:0", arg216_1: "bf16[1024][1]cuda:0", arg217_1: "bf16[1024, 1024][1024, 1]cuda:0", arg218_1: "bf16[1024][1]cuda:0", arg219_1: "bf16[1024, 1024][1024, 1]cuda:0", arg220_1: "bf16[1024][1]cuda:0", arg221_1: "bf16[1024][1]cuda:0", arg222_1: "bf16[1024][1]cuda:0", arg223_1: "bf16[4096, 1024][1024, 1]cuda:0", arg224_1: "bf16[4096][1]cuda:0", arg225_1: "bf16[1024, 4096][4096, 1]cuda:0", arg226_1: "bf16[1024][1]cuda:0", arg227_1: "bf16[1024][1]cuda:0", arg228_1: "bf16[1024][1]cuda:0", arg229_1: "bf16[1024, 1024][1024, 1]cuda:0", arg230_1: "bf16[1024][1]cuda:0", arg231_1: "bf16[1024, 1024][1024, 1]cuda:0", arg232_1: "bf16[1024][1]cuda:0", arg233_1: "bf16[1024, 1024][1024, 1]cuda:0", arg234_1: "bf16[1024][1]cuda:0", arg235_1: "bf16[1024, 1024][1024, 1]cuda:0", arg236_1: "bf16[1024][1]cuda:0", arg237_1: "bf16[1024][1]cuda:0", arg238_1: "bf16[1024][1]cuda:0", arg239_1: "bf16[4096, 1024][1024, 1]cuda:0", arg240_1: "bf16[4096][1]cuda:0", arg241_1: "bf16[1024, 4096][4096, 1]cuda:0", arg242_1: "bf16[1024][1]cuda:0", arg243_1: "bf16[1024][1]cuda:0", arg244_1: "bf16[1024][1]cuda:0", arg245_1: "bf16[1024, 1024][1024, 1]cuda:0", arg246_1: "bf16[1024][1]cuda:0", arg247_1: "bf16[1024, 1024][1024, 1]cuda:0", arg248_1: "bf16[1024][1]cuda:0", arg249_1: "bf16[1024, 1024][1024, 1]cuda:0", arg250_1: "bf16[1024][1]cuda:0", arg251_1: "bf16[1024, 1024][1024, 1]cuda:0", arg252_1: "bf16[1024][1]cuda:0", arg253_1: "bf16[1024][1]cuda:0", arg254_1: "bf16[1024][1]cuda:0", arg255_1: "bf16[4096, 1024][1024, 1]cuda:0", arg256_1: "bf16[4096][1]cuda:0", arg257_1: "bf16[1024, 4096][4096, 1]cuda:0", arg258_1: "bf16[1024][1]cuda:0", arg259_1: "bf16[1024][1]cuda:0", arg260_1: "bf16[1024][1]cuda:0", arg261_1: "bf16[1024, 1024][1024, 1]cuda:0", arg262_1: "bf16[1024][1]cuda:0", arg263_1: "bf16[1024, 1024][1024, 1]cuda:0", arg264_1: "bf16[1024][1]cuda:0", arg265_1: "bf16[1024, 1024][1024, 1]cuda:0", arg266_1: "bf16[1024][1]cuda:0", arg267_1: "bf16[1024, 1024][1024, 1]cuda:0", arg268_1: "bf16[1024][1]cuda:0", arg269_1: "bf16[1024][1]cuda:0", arg270_1: "bf16[1024][1]cuda:0", arg271_1: "bf16[4096, 1024][1024, 1]cuda:0", arg272_1: "bf16[4096][1]cuda:0", arg273_1: "bf16[1024, 4096][4096, 1]cuda:0", arg274_1: "bf16[1024][1]cuda:0", arg275_1: "bf16[1024][1]cuda:0", arg276_1: "bf16[1024][1]cuda:0", arg277_1: "bf16[1024, 1024][1024, 1]cuda:0", arg278_1: "bf16[1024][1]cuda:0", arg279_1: "bf16[1024, 1024][1024, 1]cuda:0", arg280_1: "bf16[1024][1]cuda:0", arg281_1: "bf16[1024, 1024][1024, 1]cuda:0", arg282_1: "bf16[1024][1]cuda:0", arg283_1: "bf16[1024, 1024][1024, 1]cuda:0", arg284_1: "bf16[1024][1]cuda:0", arg285_1: "bf16[1024][1]cuda:0", arg286_1: "bf16[1024][1]cuda:0", arg287_1: "bf16[4096, 1024][1024, 1]cuda:0", arg288_1: "bf16[4096][1]cuda:0", arg289_1: "bf16[1024, 4096][4096, 1]cuda:0", arg290_1: "bf16[1024][1]cuda:0", arg291_1: "bf16[1024][1]cuda:0", arg292_1: "bf16[1024][1]cuda:0", arg293_1: "bf16[1024, 1024][1024, 1]cuda:0", arg294_1: "bf16[1024][1]cuda:0", arg295_1: "bf16[1024, 1024][1024, 1]cuda:0", arg296_1: "bf16[1024][1]cuda:0", arg297_1: "bf16[1024, 1024][1024, 1]cuda:0", arg298_1: "bf16[1024][1]cuda:0", arg299_1: "bf16[1024, 1024][1024, 1]cuda:0", arg300_1: "bf16[1024][1]cuda:0", arg301_1: "bf16[1024][1]cuda:0", arg302_1: "bf16[1024][1]cuda:0", arg303_1: "bf16[4096, 1024][1024, 1]cuda:0", arg304_1: "bf16[4096][1]cuda:0", arg305_1: "bf16[1024, 4096][4096, 1]cuda:0", arg306_1: "bf16[1024][1]cuda:0", arg307_1: "bf16[1024][1]cuda:0", arg308_1: "bf16[1024][1]cuda:0", arg309_1: "bf16[1024, 1024][1024, 1]cuda:0", arg310_1: "bf16[1024][1]cuda:0", arg311_1: "bf16[1024, 1024][1024, 1]cuda:0", arg312_1: "bf16[1024][1]cuda:0", arg313_1: "bf16[1024, 1024][1024, 1]cuda:0", arg314_1: "bf16[1024][1]cuda:0", arg315_1: "bf16[1024, 1024][1024, 1]cuda:0", arg316_1: "bf16[1024][1]cuda:0", arg317_1: "bf16[1024][1]cuda:0", arg318_1: "bf16[1024][1]cuda:0", arg319_1: "bf16[4096, 1024][1024, 1]cuda:0", arg320_1: "bf16[4096][1]cuda:0", arg321_1: "bf16[1024, 4096][4096, 1]cuda:0", arg322_1: "bf16[1024][1]cuda:0", arg323_1: "bf16[1024][1]cuda:0", arg324_1: "bf16[1024][1]cuda:0", arg325_1: "bf16[1024, 1024][1024, 1]cuda:0", arg326_1: "bf16[1024][1]cuda:0", arg327_1: "bf16[1024, 1024][1024, 1]cuda:0", arg328_1: "bf16[1024][1]cuda:0", arg329_1: "bf16[1024, 1024][1024, 1]cuda:0", arg330_1: "bf16[1024][1]cuda:0", arg331_1: "bf16[1024, 1024][1024, 1]cuda:0", arg332_1: "bf16[1024][1]cuda:0", arg333_1: "bf16[1024][1]cuda:0", arg334_1: "bf16[1024][1]cuda:0", arg335_1: "bf16[4096, 1024][1024, 1]cuda:0", arg336_1: "bf16[4096][1]cuda:0", arg337_1: "bf16[1024, 4096][4096, 1]cuda:0", arg338_1: "bf16[1024][1]cuda:0", arg339_1: "bf16[1024][1]cuda:0", arg340_1: "bf16[1024][1]cuda:0", arg341_1: "bf16[1024, 1024][1024, 1]cuda:0", arg342_1: "bf16[1024][1]cuda:0", arg343_1: "bf16[1024, 1024][1024, 1]cuda:0", arg344_1: "bf16[1024][1]cuda:0", arg345_1: "bf16[1024, 1024][1024, 1]cuda:0", arg346_1: "bf16[1024][1]cuda:0", arg347_1: "bf16[1024, 1024][1024, 1]cuda:0", arg348_1: "bf16[1024][1]cuda:0", arg349_1: "bf16[1024][1]cuda:0", arg350_1: "bf16[1024][1]cuda:0", arg351_1: "bf16[4096, 1024][1024, 1]cuda:0", arg352_1: "bf16[4096][1]cuda:0", arg353_1: "bf16[1024, 4096][4096, 1]cuda:0", arg354_1: "bf16[1024][1]cuda:0", arg355_1: "bf16[1024][1]cuda:0", arg356_1: "bf16[1024][1]cuda:0", arg357_1: "bf16[1024, 1024][1024, 1]cuda:0", arg358_1: "bf16[1024][1]cuda:0", arg359_1: "bf16[1024, 1024][1024, 1]cuda:0", arg360_1: "bf16[1024][1]cuda:0", arg361_1: "bf16[1024, 1024][1024, 1]cuda:0", arg362_1: "bf16[1024][1]cuda:0", arg363_1: "bf16[1024, 1024][1024, 1]cuda:0", arg364_1: "bf16[1024][1]cuda:0", arg365_1: "bf16[1024][1]cuda:0", arg366_1: "bf16[1024][1]cuda:0", arg367_1: "bf16[4096, 1024][1024, 1]cuda:0", arg368_1: "bf16[4096][1]cuda:0", arg369_1: "bf16[1024, 4096][4096, 1]cuda:0", arg370_1: "bf16[1024][1]cuda:0", arg371_1: "bf16[1024][1]cuda:0", arg372_1: "bf16[1024][1]cuda:0", arg373_1: "bf16[1024, 1024][1024, 1]cuda:0", arg374_1: "bf16[1024][1]cuda:0", arg375_1: "bf16[1024, 1024][1024, 1]cuda:0", arg376_1: "bf16[1024][1]cuda:0", arg377_1: "bf16[1024, 1024][1024, 1]cuda:0", arg378_1: "bf16[1024][1]cuda:0", arg379_1: "bf16[1024, 1024][1024, 1]cuda:0", arg380_1: "bf16[1024][1]cuda:0", arg381_1: "bf16[1024][1]cuda:0", arg382_1: "bf16[1024][1]cuda:0", arg383_1: "bf16[4096, 1024][1024, 1]cuda:0", arg384_1: "bf16[4096][1]cuda:0", arg385_1: "bf16[1024, 4096][4096, 1]cuda:0", arg386_1: "bf16[1024][1]cuda:0", arg387_1: "bf16[1024][1]cuda:0", arg388_1: "bf16[1024][1]cuda:0", arg389_1: "i64[32, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "bf16[][]cuda:0" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:50 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg0_1 = None
        mul: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:441 in forward, code: position_ids = torch.arange(
        iota_4: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:447 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_6: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:96 in forward, code: position_ids = position_ids + self.offset
        add_2: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_6, 2);  unsqueeze_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:102 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        view: "i64[128][1]cuda:0" = torch.ops.aten.reshape.default(add_2, [-1]);  add_2 = None
        index: "bf16[128, 1024][1024, 1]cuda:0" = torch.ops.aten.index.Tensor(arg2_1, [view]);  arg2_1 = view = None
        view_1: "bf16[1, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(index, [1, 128, 1024]);  index = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:458 in forward, code: hidden_states = inputs_embeds + self.embed_positions(position_ids, past_key_values_length).to(
        add_3: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, view_1);  mul = view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant1: "f32[][]cuda:0" = self._tensor_constant1;  _tensor_constant1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_1: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_5: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_2: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 1024])
        permute: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg6_1, view_2, permute);  arg6_1 = view_2 = permute = None
        view_3: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [32, 128, 1024]);  addmm = None
        mul_3: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_10: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_3, [32, 128, 16, 64]);  mul_3 = None
        permute_5: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_1: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [512, 128, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_4: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 1024])
        permute_1: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg8_1, view_4, permute_1);  arg8_1 = view_4 = permute_1 = None
        view_5: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 128, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_8: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [32, 128, -1, 64]);  view_5 = None
        permute_3: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_2: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_12: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [512, 128, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_6: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_14: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 16, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_3: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_3: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_4: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2);  unsqueeze_5 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[32, 1, 128, 128][0, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(le, [32, -1, 128, 128]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        add_6: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_14, where);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_2: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_6, full_default_2);  add_6 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_15: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum, [512, 128, 128]);  maximum = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_13: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        amax: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_1: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax);  convert_element_type_13 = amax = None
        exp: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_14: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_6: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 1024]);  convert_element_type_1 = None
        permute_2: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg10_1, view_6, permute_2);  arg10_1 = view_6 = permute_2 = None
        view_7: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 128, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_9: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_7, [32, 128, -1, 64]);  view_7 = None
        permute_4: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_3: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_13: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [512, 128, 64]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_1: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_14, view_13);  convert_element_type_14 = view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_18: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [32, 16, 128, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_7: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_5: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_19: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [32, 128, 1024]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_20: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_19, [4096, 1024]);  view_19 = None
        permute_8: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view_20, permute_8);  arg12_1 = view_20 = permute_8 = None
        view_21: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [32, 128, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_7: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3, view_21);  add_3 = view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_20: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_3);  convert_element_type_20 = getitem_3 = None
        add_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_4: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        add_9: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        convert_element_type_21: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_22: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_21, [4096, 1024]);  convert_element_type_21 = None
        permute_9: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, view_22, permute_9);  arg16_1 = view_22 = permute_9 = None
        view_23: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_25: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.float32);  view_23 = None
        mul_6: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, 0.5)
        mul_7: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, 0.7071067811865476);  convert_element_type_25 = None
        erf: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_10: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, add_10);  mul_6 = add_10 = None
        convert_element_type_26: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_24: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_26, [4096, 4096]);  convert_element_type_26 = None
        permute_10: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg18_1, view_24, permute_10);  arg18_1 = view_24 = permute_10 = None
        view_25: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 128, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_11: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_7, view_25);  add_7 = view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_30: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant2: "f32[][]cuda:0" = self._tensor_constant2;  _tensor_constant2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_3: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_5);  convert_element_type_30 = getitem_5 = None
        add_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
        add_13: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        convert_element_type_31: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_26: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [4096, 1024])
        permute_11: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg22_1, view_26, permute_11);  arg22_1 = view_26 = permute_11 = None
        view_27: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [32, 128, 1024]);  addmm_6 = None
        mul_11: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_27, 0.125);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_34: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_11, [32, 128, 16, 64]);  mul_11 = None
        permute_16: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_9: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_35: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [512, 128, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_28: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [4096, 1024])
        permute_12: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg24_1, view_28, permute_12);  arg24_1 = view_28 = permute_12 = None
        view_29: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 128, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_32: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [32, 128, -1, 64]);  view_29 = None
        permute_14: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_10: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_36: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [512, 128, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_17: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        bmm_2: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_35, permute_17);  view_35 = permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_38: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [32, 16, 128, 128]);  bmm_2 = None
        add_14: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_38, where);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_3: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_1: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_14, full_default_3);  add_14 = full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_39: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_1, [512, 128, 128]);  maximum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_43: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        amax_1: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_43, [-1], True)
        sub_4: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_43, amax_1);  convert_element_type_43 = amax_1 = None
        exp_1: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_44: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_30: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [4096, 1024]);  convert_element_type_31 = None
        permute_13: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg26_1, view_30, permute_13);  arg26_1 = view_30 = permute_13 = None
        view_31: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [32, 128, 1024]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_33: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_31, [32, 128, -1, 64]);  view_31 = None
        permute_15: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_11: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_37: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [512, 128, 64]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_3: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_44, view_37);  convert_element_type_44 = view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_42: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [32, 16, 128, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_18: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_13: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_43: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [32, 128, 1024]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_44: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_43, [4096, 1024]);  view_43 = None
        permute_19: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg28_1, view_44, permute_19);  arg28_1 = view_44 = permute_19 = None
        view_45: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [32, 128, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_15: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, view_45);  add_11 = view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_50: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_7);  convert_element_type_50 = getitem_7 = None
        add_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_12: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
        add_17: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        convert_element_type_51: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_46: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_51, [4096, 1024]);  convert_element_type_51 = None
        permute_20: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, view_46, permute_20);  arg32_1 = view_46 = permute_20 = None
        view_47: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 128, 4096]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_55: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        mul_14: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.5)
        mul_15: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.7071067811865476);  convert_element_type_55 = None
        erf_1: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_18: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, add_18);  mul_14 = add_18 = None
        convert_element_type_56: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_48: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_56, [4096, 4096]);  convert_element_type_56 = None
        permute_21: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg34_1, view_48, permute_21);  arg34_1 = view_48 = permute_21 = None
        view_49: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [32, 128, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_19: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_15, view_49);  add_15 = view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_60: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant3: "f32[][]cuda:0" = self._tensor_constant3;  _tensor_constant3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_6: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_9);  convert_element_type_60 = getitem_9 = None
        add_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_17: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_18: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
        add_21: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        convert_element_type_61: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_50: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [4096, 1024])
        permute_22: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg38_1, view_50, permute_22);  arg38_1 = view_50 = permute_22 = None
        view_51: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [32, 128, 1024]);  addmm_12 = None
        mul_19: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_51, 0.125);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_58: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [32, 128, 16, 64]);  mul_19 = None
        permute_27: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_17: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_59: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [512, 128, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_52: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [4096, 1024])
        permute_23: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg40_1, view_52, permute_23);  arg40_1 = view_52 = permute_23 = None
        view_53: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 128, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_56: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_53, [32, 128, -1, 64]);  view_53 = None
        permute_25: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_18: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_60: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [512, 128, 64]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_28: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        bmm_4: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_59, permute_28);  view_59 = permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_62: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [32, 16, 128, 128]);  bmm_4 = None
        add_22: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_62, where);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_4: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_2: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_22, full_default_4);  add_22 = full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_63: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_2, [512, 128, 128]);  maximum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_73: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        amax_2: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_73, [-1], True)
        sub_7: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_73, amax_2);  convert_element_type_73 = amax_2 = None
        exp_2: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_74: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_54: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [4096, 1024]);  convert_element_type_61 = None
        permute_24: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_54, permute_24);  arg42_1 = view_54 = permute_24 = None
        view_55: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [32, 128, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_57: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_55, [32, 128, -1, 64]);  view_55 = None
        permute_26: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_19: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_61: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [512, 128, 64]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_5: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_74, view_61);  convert_element_type_74 = view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_66: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [32, 16, 128, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_29: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_21: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_67: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [32, 128, 1024]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_68: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [4096, 1024]);  view_67 = None
        permute_30: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_68, permute_30);  arg44_1 = view_68 = permute_30 = None
        view_69: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [32, 128, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_23: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_19, view_69);  add_19 = view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_80: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_11);  convert_element_type_80 = getitem_11 = None
        add_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_20: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_21: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
        add_25: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
        convert_element_type_81: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_70: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_81, [4096, 1024]);  convert_element_type_81 = None
        permute_31: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, view_70, permute_31);  arg48_1 = view_70 = permute_31 = None
        view_71: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 128, 4096]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_85: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_71, torch.float32);  view_71 = None
        mul_22: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.5)
        mul_23: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.7071067811865476);  convert_element_type_85 = None
        erf_2: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_26: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, add_26);  mul_22 = add_26 = None
        convert_element_type_86: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_72: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_86, [4096, 4096]);  convert_element_type_86 = None
        permute_32: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg50_1, view_72, permute_32);  arg50_1 = view_72 = permute_32 = None
        view_73: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [32, 128, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_27: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_23, view_73);  add_23 = view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_90: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant4: "f32[][]cuda:0" = self._tensor_constant4;  _tensor_constant4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_9: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_13);  convert_element_type_90 = getitem_13 = None
        add_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_25: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_26: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
        add_29: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
        convert_element_type_91: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_74: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [4096, 1024])
        permute_33: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg54_1, view_74, permute_33);  arg54_1 = view_74 = permute_33 = None
        view_75: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [32, 128, 1024]);  addmm_18 = None
        mul_27: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_75, 0.125);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_82: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [32, 128, 16, 64]);  mul_27 = None
        permute_38: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_25: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_83: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [512, 128, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_76: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [4096, 1024])
        permute_34: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg56_1, view_76, permute_34);  arg56_1 = view_76 = permute_34 = None
        view_77: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 128, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_80: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_77, [32, 128, -1, 64]);  view_77 = None
        permute_36: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_26: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_84: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [512, 128, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_39: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_84, [0, 2, 1]);  view_84 = None
        bmm_6: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_83, permute_39);  view_83 = permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_86: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [32, 16, 128, 128]);  bmm_6 = None
        add_30: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_86, where);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_5: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_3: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_30, full_default_5);  add_30 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_87: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_3, [512, 128, 128]);  maximum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_103: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_87, torch.float32);  view_87 = None
        amax_3: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_103, [-1], True)
        sub_10: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_103, amax_3);  convert_element_type_103 = amax_3 = None
        exp_3: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_104: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_78: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [4096, 1024]);  convert_element_type_91 = None
        permute_35: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg58_1, view_78, permute_35);  arg58_1 = view_78 = permute_35 = None
        view_79: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [32, 128, 1024]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_81: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_79, [32, 128, -1, 64]);  view_79 = None
        permute_37: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_27: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_85: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [512, 128, 64]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_7: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_104, view_85);  convert_element_type_104 = view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_90: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [32, 16, 128, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_40: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_29: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_91: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [32, 128, 1024]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_92: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_91, [4096, 1024]);  view_91 = None
        permute_41: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg60_1, view_92, permute_41);  arg60_1 = view_92 = permute_41 = None
        view_93: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [32, 128, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_31: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_27, view_93);  add_27 = view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_110: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_15);  convert_element_type_110 = getitem_15 = None
        add_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_28: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_29: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
        add_33: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
        convert_element_type_111: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_94: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_111, [4096, 1024]);  convert_element_type_111 = None
        permute_42: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg64_1, view_94, permute_42);  arg64_1 = view_94 = permute_42 = None
        view_95: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 128, 4096]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_115: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_95, torch.float32);  view_95 = None
        mul_30: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5)
        mul_31: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.7071067811865476);  convert_element_type_115 = None
        erf_3: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_34: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, add_34);  mul_30 = add_34 = None
        convert_element_type_116: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_96: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [4096, 4096]);  convert_element_type_116 = None
        permute_43: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg66_1, view_96, permute_43);  arg66_1 = view_96 = permute_43 = None
        view_97: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [32, 128, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_35: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_31, view_97);  add_31 = view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_120: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant5: "f32[][]cuda:0" = self._tensor_constant5;  _tensor_constant5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_12: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_17);  convert_element_type_120 = getitem_17 = None
        add_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_33: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_34: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
        add_37: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
        convert_element_type_121: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_98: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [4096, 1024])
        permute_44: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg70_1, view_98, permute_44);  arg70_1 = view_98 = permute_44 = None
        view_99: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [32, 128, 1024]);  addmm_24 = None
        mul_35: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_99, 0.125);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_106: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_35, [32, 128, 16, 64]);  mul_35 = None
        permute_49: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_33: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_107: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [512, 128, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_100: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [4096, 1024])
        permute_45: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_100, permute_45);  arg72_1 = view_100 = permute_45 = None
        view_101: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 128, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_104: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_101, [32, 128, -1, 64]);  view_101 = None
        permute_47: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_34: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_108: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [512, 128, 64]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_50: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
        bmm_8: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_107, permute_50);  view_107 = permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_110: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [32, 16, 128, 128]);  bmm_8 = None
        add_38: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_110, where);  view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_6: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_4: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_38, full_default_6);  add_38 = full_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_111: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_4, [512, 128, 128]);  maximum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_133: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_111, torch.float32);  view_111 = None
        amax_4: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_133, [-1], True)
        sub_13: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, amax_4);  convert_element_type_133 = amax_4 = None
        exp_4: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_134: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_102: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [4096, 1024]);  convert_element_type_121 = None
        permute_46: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_102, permute_46);  arg74_1 = view_102 = permute_46 = None
        view_103: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [32, 128, 1024]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_105: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [32, 128, -1, 64]);  view_103 = None
        permute_48: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_35: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_109: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [512, 128, 64]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_9: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_134, view_109);  convert_element_type_134 = view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_114: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [32, 16, 128, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_51: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_37: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_115: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [32, 128, 1024]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_116: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_115, [4096, 1024]);  view_115 = None
        permute_52: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg76_1, view_116, permute_52);  arg76_1 = view_116 = permute_52 = None
        view_117: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [32, 128, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_39: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_35, view_117);  add_35 = view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_140: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_140, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_140, getitem_19);  convert_element_type_140 = getitem_19 = None
        add_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_36: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_37: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
        add_41: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
        convert_element_type_141: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_118: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_141, [4096, 1024]);  convert_element_type_141 = None
        permute_53: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_118, permute_53);  arg80_1 = view_118 = permute_53 = None
        view_119: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 128, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_145: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        mul_38: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, 0.5)
        mul_39: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, 0.7071067811865476);  convert_element_type_145 = None
        erf_4: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_42: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_42);  mul_38 = add_42 = None
        convert_element_type_146: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_120: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_146, [4096, 4096]);  convert_element_type_146 = None
        permute_54: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg82_1, view_120, permute_54);  arg82_1 = view_120 = permute_54 = None
        view_121: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [32, 128, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_43: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_39, view_121);  add_39 = view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_150: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_150, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant6: "f32[][]cuda:0" = self._tensor_constant6;  _tensor_constant6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_15: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_150, getitem_21);  convert_element_type_150 = getitem_21 = None
        add_44: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_41: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_42: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
        add_45: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
        convert_element_type_151: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_122: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [4096, 1024])
        permute_55: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg86_1, view_122, permute_55);  arg86_1 = view_122 = permute_55 = None
        view_123: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [32, 128, 1024]);  addmm_30 = None
        mul_43: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_123, 0.125);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_130: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_43, [32, 128, 16, 64]);  mul_43 = None
        permute_60: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_41: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_131: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [512, 128, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_124: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [4096, 1024])
        permute_56: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg88_1, view_124, permute_56);  arg88_1 = view_124 = permute_56 = None
        view_125: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 128, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_128: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [32, 128, -1, 64]);  view_125 = None
        permute_58: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_42: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_132: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [512, 128, 64]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_61: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
        bmm_10: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_131, permute_61);  view_131 = permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_134: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [32, 16, 128, 128]);  bmm_10 = None
        add_46: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_134, where);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_7: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_5: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_46, full_default_7);  add_46 = full_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_135: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_5, [512, 128, 128]);  maximum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_163: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_135, torch.float32);  view_135 = None
        amax_5: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_163, [-1], True)
        sub_16: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, amax_5);  convert_element_type_163 = amax_5 = None
        exp_5: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_164: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_126: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [4096, 1024]);  convert_element_type_151 = None
        permute_57: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg90_1, view_126, permute_57);  arg90_1 = view_126 = permute_57 = None
        view_127: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [32, 128, 1024]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_129: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_127, [32, 128, -1, 64]);  view_127 = None
        permute_59: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_43: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_133: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [512, 128, 64]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_11: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_164, view_133);  convert_element_type_164 = view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_138: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [32, 16, 128, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_62: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_45: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_139: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [32, 128, 1024]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_140: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_139, [4096, 1024]);  view_139 = None
        permute_63: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg92_1, view_140, permute_63);  arg92_1 = view_140 = permute_63 = None
        view_141: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [32, 128, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_47: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_43, view_141);  add_43 = view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_170: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_23);  convert_element_type_170 = getitem_23 = None
        add_48: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_44: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_45: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
        add_49: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
        convert_element_type_171: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_142: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_171, [4096, 1024]);  convert_element_type_171 = None
        permute_64: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg96_1, view_142, permute_64);  arg96_1 = view_142 = permute_64 = None
        view_143: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 128, 4096]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_175: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        mul_46: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.5)
        mul_47: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.7071067811865476);  convert_element_type_175 = None
        erf_5: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_50: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, add_50);  mul_46 = add_50 = None
        convert_element_type_176: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_144: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_176, [4096, 4096]);  convert_element_type_176 = None
        permute_65: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, view_144, permute_65);  arg98_1 = view_144 = permute_65 = None
        view_145: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [32, 128, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_51: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_47, view_145);  add_47 = view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_180: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_180, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant7: "f32[][]cuda:0" = self._tensor_constant7;  _tensor_constant7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_18: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_180, getitem_25);  convert_element_type_180 = getitem_25 = None
        add_52: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_49: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_50: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
        add_53: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
        convert_element_type_181: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_146: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_181, [4096, 1024])
        permute_66: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg102_1, view_146, permute_66);  arg102_1 = view_146 = permute_66 = None
        view_147: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [32, 128, 1024]);  addmm_36 = None
        mul_51: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_147, 0.125);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_154: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_51, [32, 128, 16, 64]);  mul_51 = None
        permute_71: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_49: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_155: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [512, 128, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_148: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_181, [4096, 1024])
        permute_67: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg104_1, view_148, permute_67);  arg104_1 = view_148 = permute_67 = None
        view_149: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 128, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_152: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_149, [32, 128, -1, 64]);  view_149 = None
        permute_69: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_50: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_156: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [512, 128, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_72: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None
        bmm_12: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_155, permute_72);  view_155 = permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_158: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [32, 16, 128, 128]);  bmm_12 = None
        add_54: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_158, where);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_8: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_6: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_54, full_default_8);  add_54 = full_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_159: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_6, [512, 128, 128]);  maximum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_193: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        amax_6: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_193, [-1], True)
        sub_19: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_193, amax_6);  convert_element_type_193 = amax_6 = None
        exp_6: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_194: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_150: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_181, [4096, 1024]);  convert_element_type_181 = None
        permute_68: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg106_1, view_150, permute_68);  arg106_1 = view_150 = permute_68 = None
        view_151: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [32, 128, 1024]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_153: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_151, [32, 128, -1, 64]);  view_151 = None
        permute_70: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_51: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_157: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [512, 128, 64]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_13: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_194, view_157);  convert_element_type_194 = view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_162: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [32, 16, 128, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_73: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_53: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_163: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [32, 128, 1024]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_164: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_163, [4096, 1024]);  view_163 = None
        permute_74: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg108_1, view_164, permute_74);  arg108_1 = view_164 = permute_74 = None
        view_165: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [32, 128, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_55: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_51, view_165);  add_51 = view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_200: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_200, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_200, getitem_27);  convert_element_type_200 = getitem_27 = None
        add_56: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_52: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_53: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, arg109_1);  mul_52 = arg109_1 = None
        add_57: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
        convert_element_type_201: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_166: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_201, [4096, 1024]);  convert_element_type_201 = None
        permute_75: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg112_1, view_166, permute_75);  arg112_1 = view_166 = permute_75 = None
        view_167: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 128, 4096]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_205: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_167, torch.float32);  view_167 = None
        mul_54: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.5)
        mul_55: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.7071067811865476);  convert_element_type_205 = None
        erf_6: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_58: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, add_58);  mul_54 = add_58 = None
        convert_element_type_206: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_168: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_206, [4096, 4096]);  convert_element_type_206 = None
        permute_76: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg114_1, view_168, permute_76);  arg114_1 = view_168 = permute_76 = None
        view_169: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [32, 128, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_59: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_55, view_169);  add_55 = view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_210: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_210, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant8: "f32[][]cuda:0" = self._tensor_constant8;  _tensor_constant8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_21: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_210, getitem_29);  convert_element_type_210 = getitem_29 = None
        add_60: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_57: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_58: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg115_1);  mul_57 = arg115_1 = None
        add_61: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
        convert_element_type_211: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_170: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_211, [4096, 1024])
        permute_77: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg118_1, view_170, permute_77);  arg118_1 = view_170 = permute_77 = None
        view_171: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [32, 128, 1024]);  addmm_42 = None
        mul_59: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_171, 0.125);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_178: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_59, [32, 128, 16, 64]);  mul_59 = None
        permute_82: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_57: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_179: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [512, 128, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_172: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_211, [4096, 1024])
        permute_78: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg120_1, view_172, permute_78);  arg120_1 = view_172 = permute_78 = None
        view_173: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [32, 128, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_176: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_173, [32, 128, -1, 64]);  view_173 = None
        permute_80: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_58: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_180: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [512, 128, 64]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_83: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_180, [0, 2, 1]);  view_180 = None
        bmm_14: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_179, permute_83);  view_179 = permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_182: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [32, 16, 128, 128]);  bmm_14 = None
        add_62: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_182, where);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_9: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_7: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_62, full_default_9);  add_62 = full_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_183: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_7, [512, 128, 128]);  maximum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_223: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_183, torch.float32);  view_183 = None
        amax_7: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_223, [-1], True)
        sub_22: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_223, amax_7);  convert_element_type_223 = amax_7 = None
        exp_7: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_224: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_174: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_211, [4096, 1024]);  convert_element_type_211 = None
        permute_79: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg122_1, view_174, permute_79);  arg122_1 = view_174 = permute_79 = None
        view_175: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [32, 128, 1024]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_177: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_175, [32, 128, -1, 64]);  view_175 = None
        permute_81: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_59: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_181: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [512, 128, 64]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_15: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_224, view_181);  convert_element_type_224 = view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_186: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [32, 16, 128, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_84: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_61: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_187: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [32, 128, 1024]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_188: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_187, [4096, 1024]);  view_187 = None
        permute_85: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg124_1, view_188, permute_85);  arg124_1 = view_188 = permute_85 = None
        view_189: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [32, 128, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_63: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_59, view_189);  add_59 = view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_230: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_230, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, getitem_31);  convert_element_type_230 = getitem_31 = None
        add_64: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_60: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_61: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, arg125_1);  mul_60 = arg125_1 = None
        add_65: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
        convert_element_type_231: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_190: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_231, [4096, 1024]);  convert_element_type_231 = None
        permute_86: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg128_1, view_190, permute_86);  arg128_1 = view_190 = permute_86 = None
        view_191: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 128, 4096]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_235: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_191, torch.float32);  view_191 = None
        mul_62: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.5)
        mul_63: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.7071067811865476);  convert_element_type_235 = None
        erf_7: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_66: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, add_66);  mul_62 = add_66 = None
        convert_element_type_236: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_192: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [4096, 4096]);  convert_element_type_236 = None
        permute_87: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg130_1, view_192, permute_87);  arg130_1 = view_192 = permute_87 = None
        view_193: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [32, 128, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_67: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_63, view_193);  add_63 = view_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_240: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_240, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant9: "f32[][]cuda:0" = self._tensor_constant9;  _tensor_constant9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_24: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, getitem_33);  convert_element_type_240 = getitem_33 = None
        add_68: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_65: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_66: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
        add_69: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
        convert_element_type_241: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_194: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [4096, 1024])
        permute_88: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg134_1, view_194, permute_88);  arg134_1 = view_194 = permute_88 = None
        view_195: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [32, 128, 1024]);  addmm_48 = None
        mul_67: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, 0.125);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_202: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [32, 128, 16, 64]);  mul_67 = None
        permute_93: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_65: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_203: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [512, 128, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_196: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [4096, 1024])
        permute_89: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg136_1, view_196, permute_89);  arg136_1 = view_196 = permute_89 = None
        view_197: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [32, 128, 1024]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_200: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_197, [32, 128, -1, 64]);  view_197 = None
        permute_91: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_66: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_204: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [512, 128, 64]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_94: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
        bmm_16: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_203, permute_94);  view_203 = permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_206: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [32, 16, 128, 128]);  bmm_16 = None
        add_70: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_206, where);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_10: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_8: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_70, full_default_10);  add_70 = full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_207: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_8, [512, 128, 128]);  maximum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_253: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_207, torch.float32);  view_207 = None
        amax_8: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_253, [-1], True)
        sub_25: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_253, amax_8);  convert_element_type_253 = amax_8 = None
        exp_8: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_254: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_198: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [4096, 1024]);  convert_element_type_241 = None
        permute_90: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg138_1, view_198, permute_90);  arg138_1 = view_198 = permute_90 = None
        view_199: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [32, 128, 1024]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_201: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [32, 128, -1, 64]);  view_199 = None
        permute_92: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_67: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_205: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [512, 128, 64]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_17: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_254, view_205);  convert_element_type_254 = view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_210: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [32, 16, 128, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_95: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_69: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_211: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [32, 128, 1024]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_212: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_211, [4096, 1024]);  view_211 = None
        permute_96: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_212, permute_96);  arg140_1 = view_212 = permute_96 = None
        view_213: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [32, 128, 1024]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_71: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_67, view_213);  add_67 = view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_260: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_260, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_260, getitem_35);  convert_element_type_260 = getitem_35 = None
        add_72: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_68: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_69: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, arg141_1);  mul_68 = arg141_1 = None
        add_73: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, arg142_1);  mul_69 = arg142_1 = None
        convert_element_type_261: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_214: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_261, [4096, 1024]);  convert_element_type_261 = None
        permute_97: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg144_1, view_214, permute_97);  arg144_1 = view_214 = permute_97 = None
        view_215: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 128, 4096]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_265: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_215, torch.float32);  view_215 = None
        mul_70: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, 0.5)
        mul_71: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, 0.7071067811865476);  convert_element_type_265 = None
        erf_8: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_74: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, add_74);  mul_70 = add_74 = None
        convert_element_type_266: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_216: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_266, [4096, 4096]);  convert_element_type_266 = None
        permute_98: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg146_1, view_216, permute_98);  arg146_1 = view_216 = permute_98 = None
        view_217: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [32, 128, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_75: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_71, view_217);  add_71 = view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_270: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_270, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant10: "f32[][]cuda:0" = self._tensor_constant10;  _tensor_constant10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_27: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_270, getitem_37);  convert_element_type_270 = getitem_37 = None
        add_76: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_73: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_74: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg147_1);  mul_73 = arg147_1 = None
        add_77: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg148_1);  mul_74 = arg148_1 = None
        convert_element_type_271: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_218: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_271, [4096, 1024])
        permute_99: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg150_1, view_218, permute_99);  arg150_1 = view_218 = permute_99 = None
        view_219: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [32, 128, 1024]);  addmm_54 = None
        mul_75: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_219, 0.125);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_226: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [32, 128, 16, 64]);  mul_75 = None
        permute_104: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_73: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_227: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [512, 128, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_220: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_271, [4096, 1024])
        permute_100: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg152_1, view_220, permute_100);  arg152_1 = view_220 = permute_100 = None
        view_221: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [32, 128, 1024]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_224: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [32, 128, -1, 64]);  view_221 = None
        permute_102: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_74: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_228: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [512, 128, 64]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_105: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
        bmm_18: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_227, permute_105);  view_227 = permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_230: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [32, 16, 128, 128]);  bmm_18 = None
        add_78: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_230, where);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_11: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_9: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_78, full_default_11);  add_78 = full_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_231: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_9, [512, 128, 128]);  maximum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_283: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        amax_9: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_283, [-1], True)
        sub_28: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_283, amax_9);  convert_element_type_283 = amax_9 = None
        exp_9: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_284: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_222: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_271, [4096, 1024]);  convert_element_type_271 = None
        permute_101: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg154_1, view_222, permute_101);  arg154_1 = view_222 = permute_101 = None
        view_223: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [32, 128, 1024]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_225: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_223, [32, 128, -1, 64]);  view_223 = None
        permute_103: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_75: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_229: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [512, 128, 64]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_19: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_284, view_229);  convert_element_type_284 = view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_234: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [32, 16, 128, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_106: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_77: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_235: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [32, 128, 1024]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_236: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [4096, 1024]);  view_235 = None
        permute_107: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg156_1, view_236, permute_107);  arg156_1 = view_236 = permute_107 = None
        view_237: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [32, 128, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_79: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_75, view_237);  add_75 = view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_290: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_290, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_290, getitem_39);  convert_element_type_290 = getitem_39 = None
        add_80: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_76: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_77: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, arg157_1);  mul_76 = arg157_1 = None
        add_81: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_77, arg158_1);  mul_77 = arg158_1 = None
        convert_element_type_291: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_238: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_291, [4096, 1024]);  convert_element_type_291 = None
        permute_108: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_238, permute_108);  arg160_1 = view_238 = permute_108 = None
        view_239: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 128, 4096]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_295: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_78: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_295, 0.5)
        mul_79: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_295, 0.7071067811865476);  convert_element_type_295 = None
        erf_9: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_82: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, add_82);  mul_78 = add_82 = None
        convert_element_type_296: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_240: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_296, [4096, 4096]);  convert_element_type_296 = None
        permute_109: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg162_1, view_240, permute_109);  arg162_1 = view_240 = permute_109 = None
        view_241: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [32, 128, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_83: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_79, view_241);  add_79 = view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_300: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_300, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant11: "f32[][]cuda:0" = self._tensor_constant11;  _tensor_constant11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_30: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_300, getitem_41);  convert_element_type_300 = getitem_41 = None
        add_84: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_81: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_82: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, arg163_1);  mul_81 = arg163_1 = None
        add_85: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, arg164_1);  mul_82 = arg164_1 = None
        convert_element_type_301: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_242: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_301, [4096, 1024])
        permute_110: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg166_1, view_242, permute_110);  arg166_1 = view_242 = permute_110 = None
        view_243: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [32, 128, 1024]);  addmm_60 = None
        mul_83: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_243, 0.125);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_250: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_83, [32, 128, 16, 64]);  mul_83 = None
        permute_115: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_81: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_251: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [512, 128, 64]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_244: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_301, [4096, 1024])
        permute_111: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg168_1, view_244, permute_111);  arg168_1 = view_244 = permute_111 = None
        view_245: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 128, 1024]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_248: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [32, 128, -1, 64]);  view_245 = None
        permute_113: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_82: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_252: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [512, 128, 64]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_116: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
        bmm_20: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_251, permute_116);  view_251 = permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_254: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [32, 16, 128, 128]);  bmm_20 = None
        add_86: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_254, where);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_12: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_10: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_86, full_default_12);  add_86 = full_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_255: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_10, [512, 128, 128]);  maximum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_313: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_255, torch.float32);  view_255 = None
        amax_10: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_313, [-1], True)
        sub_31: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_313, amax_10);  convert_element_type_313 = amax_10 = None
        exp_10: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_314: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_246: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_301, [4096, 1024]);  convert_element_type_301 = None
        permute_112: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg170_1, view_246, permute_112);  arg170_1 = view_246 = permute_112 = None
        view_247: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [32, 128, 1024]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_249: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_247, [32, 128, -1, 64]);  view_247 = None
        permute_114: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_83: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_253: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [512, 128, 64]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_21: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_314, view_253);  convert_element_type_314 = view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_258: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [32, 16, 128, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_117: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_85: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_259: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [32, 128, 1024]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_260: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_259, [4096, 1024]);  view_259 = None
        permute_118: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg172_1, view_260, permute_118);  arg172_1 = view_260 = permute_118 = None
        view_261: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [32, 128, 1024]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_87: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_83, view_261);  add_83 = view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_320: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_320, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_320, getitem_43);  convert_element_type_320 = getitem_43 = None
        add_88: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_84: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_85: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, arg173_1);  mul_84 = arg173_1 = None
        add_89: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, arg174_1);  mul_85 = arg174_1 = None
        convert_element_type_321: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_262: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_321, [4096, 1024]);  convert_element_type_321 = None
        permute_119: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg176_1, view_262, permute_119);  arg176_1 = view_262 = permute_119 = None
        view_263: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 128, 4096]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_325: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        mul_86: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_325, 0.5)
        mul_87: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_325, 0.7071067811865476);  convert_element_type_325 = None
        erf_10: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_90: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, add_90);  mul_86 = add_90 = None
        convert_element_type_326: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_264: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_326, [4096, 4096]);  convert_element_type_326 = None
        permute_120: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg178_1, view_264, permute_120);  arg178_1 = view_264 = permute_120 = None
        view_265: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [32, 128, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_91: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_87, view_265);  add_87 = view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_330: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_330, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant12: "f32[][]cuda:0" = self._tensor_constant12;  _tensor_constant12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_33: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_330, getitem_45);  convert_element_type_330 = getitem_45 = None
        add_92: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_89: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_90: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, arg179_1);  mul_89 = arg179_1 = None
        add_93: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, arg180_1);  mul_90 = arg180_1 = None
        convert_element_type_331: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_266: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_331, [4096, 1024])
        permute_121: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg182_1, view_266, permute_121);  arg182_1 = view_266 = permute_121 = None
        view_267: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [32, 128, 1024]);  addmm_66 = None
        mul_91: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_267, 0.125);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_274: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_91, [32, 128, 16, 64]);  mul_91 = None
        permute_126: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_89: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_275: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [512, 128, 64]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_268: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_331, [4096, 1024])
        permute_122: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg184_1, view_268, permute_122);  arg184_1 = view_268 = permute_122 = None
        view_269: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [32, 128, 1024]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_272: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_269, [32, 128, -1, 64]);  view_269 = None
        permute_124: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_90: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_276: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [512, 128, 64]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_127: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
        bmm_22: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_275, permute_127);  view_275 = permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_278: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [32, 16, 128, 128]);  bmm_22 = None
        add_94: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_278, where);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_13: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_11: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_94, full_default_13);  add_94 = full_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_279: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_11, [512, 128, 128]);  maximum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_343: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        amax_11: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_343, [-1], True)
        sub_34: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_343, amax_11);  convert_element_type_343 = amax_11 = None
        exp_11: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_344: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_270: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_331, [4096, 1024]);  convert_element_type_331 = None
        permute_123: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg186_1, view_270, permute_123);  arg186_1 = view_270 = permute_123 = None
        view_271: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [32, 128, 1024]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_273: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_271, [32, 128, -1, 64]);  view_271 = None
        permute_125: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_91: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_277: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [512, 128, 64]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_23: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_344, view_277);  convert_element_type_344 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_282: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [32, 16, 128, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_128: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_93: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_283: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [32, 128, 1024]);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_284: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_283, [4096, 1024]);  view_283 = None
        permute_129: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg188_1, view_284, permute_129);  arg188_1 = view_284 = permute_129 = None
        view_285: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [32, 128, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_95: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_91, view_285);  add_91 = view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_350: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_350, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_350, getitem_47);  convert_element_type_350 = getitem_47 = None
        add_96: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_92: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_93: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, arg189_1);  mul_92 = arg189_1 = None
        add_97: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, arg190_1);  mul_93 = arg190_1 = None
        convert_element_type_351: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_286: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_351, [4096, 1024]);  convert_element_type_351 = None
        permute_130: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, view_286, permute_130);  arg192_1 = view_286 = permute_130 = None
        view_287: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 128, 4096]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_355: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        mul_94: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.5)
        mul_95: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.7071067811865476);  convert_element_type_355 = None
        erf_11: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_98: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, add_98);  mul_94 = add_98 = None
        convert_element_type_356: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_288: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_356, [4096, 4096]);  convert_element_type_356 = None
        permute_131: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg194_1, view_288, permute_131);  arg194_1 = view_288 = permute_131 = None
        view_289: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [32, 128, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_99: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_95, view_289);  add_95 = view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_360: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_360, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant13: "f32[][]cuda:0" = self._tensor_constant13;  _tensor_constant13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_36: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_360, getitem_49);  convert_element_type_360 = getitem_49 = None
        add_100: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_97: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_98: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, arg195_1);  mul_97 = arg195_1 = None
        add_101: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_98, arg196_1);  mul_98 = arg196_1 = None
        convert_element_type_361: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_290: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [4096, 1024])
        permute_132: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_72: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg198_1, view_290, permute_132);  arg198_1 = view_290 = permute_132 = None
        view_291: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [32, 128, 1024]);  addmm_72 = None
        mul_99: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_291, 0.125);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_298: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_99, [32, 128, 16, 64]);  mul_99 = None
        permute_137: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_97: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_299: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [512, 128, 64]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_292: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [4096, 1024])
        permute_133: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_73: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg200_1, view_292, permute_133);  arg200_1 = view_292 = permute_133 = None
        view_293: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [32, 128, 1024]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_296: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_293, [32, 128, -1, 64]);  view_293 = None
        permute_135: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_98: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_300: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [512, 128, 64]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_138: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None
        bmm_24: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_299, permute_138);  view_299 = permute_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_302: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [32, 16, 128, 128]);  bmm_24 = None
        add_102: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_302, where);  view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_14: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_12: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_102, full_default_14);  add_102 = full_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_303: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_12, [512, 128, 128]);  maximum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_373: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None
        amax_12: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_373, [-1], True)
        sub_37: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_373, amax_12);  convert_element_type_373 = amax_12 = None
        exp_12: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_374: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_294: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [4096, 1024]);  convert_element_type_361 = None
        permute_134: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_74: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg202_1, view_294, permute_134);  arg202_1 = view_294 = permute_134 = None
        view_295: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [32, 128, 1024]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_297: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [32, 128, -1, 64]);  view_295 = None
        permute_136: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_99: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
        view_301: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [512, 128, 64]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_25: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_374, view_301);  convert_element_type_374 = view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_306: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [32, 16, 128, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_139: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_101: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_307: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [32, 128, 1024]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_308: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_307, [4096, 1024]);  view_307 = None
        permute_140: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_75: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg204_1, view_308, permute_140);  arg204_1 = view_308 = permute_140 = None
        view_309: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [32, 128, 1024]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_103: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_99, view_309);  add_99 = view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_380: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_380, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        sub_38: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_380, getitem_51);  convert_element_type_380 = getitem_51 = None
        add_104: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_100: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
        mul_101: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, arg205_1);  mul_100 = arg205_1 = None
        add_105: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, arg206_1);  mul_101 = arg206_1 = None
        convert_element_type_381: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_310: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_381, [4096, 1024]);  convert_element_type_381 = None
        permute_141: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_76: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg208_1, view_310, permute_141);  arg208_1 = view_310 = permute_141 = None
        view_311: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [32, 128, 4096]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_385: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_311, torch.float32);  view_311 = None
        mul_102: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_385, 0.5)
        mul_103: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_385, 0.7071067811865476);  convert_element_type_385 = None
        erf_12: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_106: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_104: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, add_106);  mul_102 = add_106 = None
        convert_element_type_386: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_312: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_386, [4096, 4096]);  convert_element_type_386 = None
        permute_142: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_77: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg210_1, view_312, permute_142);  arg210_1 = view_312 = permute_142 = None
        view_313: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [32, 128, 1024]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_107: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_103, view_313);  add_103 = view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_390: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_390, [2], correction = 0, keepdim = True)
        getitem_52: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant14: "f32[][]cuda:0" = self._tensor_constant14;  _tensor_constant14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_39: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_390, getitem_53);  convert_element_type_390 = getitem_53 = None
        add_108: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_105: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
        mul_106: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, arg211_1);  mul_105 = arg211_1 = None
        add_109: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, arg212_1);  mul_106 = arg212_1 = None
        convert_element_type_391: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_314: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_391, [4096, 1024])
        permute_143: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_78: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg214_1, view_314, permute_143);  arg214_1 = view_314 = permute_143 = None
        view_315: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [32, 128, 1024]);  addmm_78 = None
        mul_107: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_315, 0.125);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_322: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_107, [32, 128, 16, 64]);  mul_107 = None
        permute_148: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_105: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_323: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [512, 128, 64]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_316: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_391, [4096, 1024])
        permute_144: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_79: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg216_1, view_316, permute_144);  arg216_1 = view_316 = permute_144 = None
        view_317: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [32, 128, 1024]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_320: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_317, [32, 128, -1, 64]);  view_317 = None
        permute_146: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_106: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_324: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [512, 128, 64]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_149: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
        bmm_26: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_323, permute_149);  view_323 = permute_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_326: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [32, 16, 128, 128]);  bmm_26 = None
        add_110: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_326, where);  view_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_15: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_13: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_110, full_default_15);  add_110 = full_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_327: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_13, [512, 128, 128]);  maximum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_403: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        amax_13: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_403, [-1], True)
        sub_40: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_403, amax_13);  convert_element_type_403 = amax_13 = None
        exp_13: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_404: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_318: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_391, [4096, 1024]);  convert_element_type_391 = None
        permute_145: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_80: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg218_1, view_318, permute_145);  arg218_1 = view_318 = permute_145 = None
        view_319: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [32, 128, 1024]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_321: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_319, [32, 128, -1, 64]);  view_319 = None
        permute_147: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_107: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_325: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [512, 128, 64]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_27: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_404, view_325);  convert_element_type_404 = view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_330: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [32, 16, 128, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_150: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_109: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_331: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [32, 128, 1024]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_332: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_331, [4096, 1024]);  view_331 = None
        permute_151: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_81: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg220_1, view_332, permute_151);  arg220_1 = view_332 = permute_151 = None
        view_333: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [32, 128, 1024]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_111: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_107, view_333);  add_107 = view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_410: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_410, [2], correction = 0, keepdim = True)
        getitem_54: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        sub_41: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_410, getitem_55);  convert_element_type_410 = getitem_55 = None
        add_112: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_108: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_109: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, arg221_1);  mul_108 = arg221_1 = None
        add_113: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, arg222_1);  mul_109 = arg222_1 = None
        convert_element_type_411: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.bfloat16);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_334: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_411, [4096, 1024]);  convert_element_type_411 = None
        permute_152: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_82: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg224_1, view_334, permute_152);  arg224_1 = view_334 = permute_152 = None
        view_335: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [32, 128, 4096]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_415: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None
        mul_110: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_415, 0.5)
        mul_111: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_415, 0.7071067811865476);  convert_element_type_415 = None
        erf_13: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_114: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_112: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, add_114);  mul_110 = add_114 = None
        convert_element_type_416: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_336: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_416, [4096, 4096]);  convert_element_type_416 = None
        permute_153: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_83: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg226_1, view_336, permute_153);  arg226_1 = view_336 = permute_153 = None
        view_337: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [32, 128, 1024]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_115: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_111, view_337);  add_111 = view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_420: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_420, [2], correction = 0, keepdim = True)
        getitem_56: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant15: "f32[][]cuda:0" = self._tensor_constant15;  _tensor_constant15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_42: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_420, getitem_57);  convert_element_type_420 = getitem_57 = None
        add_116: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_113: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
        mul_114: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, arg227_1);  mul_113 = arg227_1 = None
        add_117: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, arg228_1);  mul_114 = arg228_1 = None
        convert_element_type_421: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_338: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_421, [4096, 1024])
        permute_154: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_84: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg230_1, view_338, permute_154);  arg230_1 = view_338 = permute_154 = None
        view_339: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [32, 128, 1024]);  addmm_84 = None
        mul_115: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_339, 0.125);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_346: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_115, [32, 128, 16, 64]);  mul_115 = None
        permute_159: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_113: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_347: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [512, 128, 64]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_340: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_421, [4096, 1024])
        permute_155: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_85: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg232_1, view_340, permute_155);  arg232_1 = view_340 = permute_155 = None
        view_341: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_85, [32, 128, 1024]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_344: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_341, [32, 128, -1, 64]);  view_341 = None
        permute_157: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_114: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_348: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [512, 128, 64]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_160: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_348, [0, 2, 1]);  view_348 = None
        bmm_28: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_347, permute_160);  view_347 = permute_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_350: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [32, 16, 128, 128]);  bmm_28 = None
        add_118: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_350, where);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_16: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_14: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_118, full_default_16);  add_118 = full_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_351: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_14, [512, 128, 128]);  maximum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_433: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_351, torch.float32);  view_351 = None
        amax_14: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_433, [-1], True)
        sub_43: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_433, amax_14);  convert_element_type_433 = amax_14 = None
        exp_14: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_434: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_342: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_421, [4096, 1024]);  convert_element_type_421 = None
        permute_156: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_86: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg234_1, view_342, permute_156);  arg234_1 = view_342 = permute_156 = None
        view_343: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [32, 128, 1024]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_345: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_343, [32, 128, -1, 64]);  view_343 = None
        permute_158: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_115: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_349: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [512, 128, 64]);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_29: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_434, view_349);  convert_element_type_434 = view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_354: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [32, 16, 128, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_161: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_117: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_355: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [32, 128, 1024]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_356: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_355, [4096, 1024]);  view_355 = None
        permute_162: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_87: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg236_1, view_356, permute_162);  arg236_1 = view_356 = permute_162 = None
        view_357: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_87, [32, 128, 1024]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_119: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, view_357);  add_115 = view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_440: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_440, [2], correction = 0, keepdim = True)
        getitem_58: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        sub_44: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_440, getitem_59);  convert_element_type_440 = getitem_59 = None
        add_120: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_116: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_117: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, arg237_1);  mul_116 = arg237_1 = None
        add_121: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_117, arg238_1);  mul_117 = arg238_1 = None
        convert_element_type_441: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_358: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_441, [4096, 1024]);  convert_element_type_441 = None
        permute_163: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_88: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg240_1, view_358, permute_163);  arg240_1 = view_358 = permute_163 = None
        view_359: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [32, 128, 4096]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_445: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        mul_118: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_445, 0.5)
        mul_119: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_445, 0.7071067811865476);  convert_element_type_445 = None
        erf_14: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_122: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_120: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, add_122);  mul_118 = add_122 = None
        convert_element_type_446: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_360: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_446, [4096, 4096]);  convert_element_type_446 = None
        permute_164: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_89: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg242_1, view_360, permute_164);  arg242_1 = view_360 = permute_164 = None
        view_361: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_89, [32, 128, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_123: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_119, view_361);  add_119 = view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_450: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_123, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_450, [2], correction = 0, keepdim = True)
        getitem_60: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant16: "f32[][]cuda:0" = self._tensor_constant16;  _tensor_constant16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_45: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_450, getitem_61);  convert_element_type_450 = getitem_61 = None
        add_124: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_121: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
        mul_122: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, arg243_1);  mul_121 = arg243_1 = None
        add_125: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, arg244_1);  mul_122 = arg244_1 = None
        convert_element_type_451: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_362: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_451, [4096, 1024])
        permute_165: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_90: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg246_1, view_362, permute_165);  arg246_1 = view_362 = permute_165 = None
        view_363: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [32, 128, 1024]);  addmm_90 = None
        mul_123: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_363, 0.125);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_370: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_123, [32, 128, 16, 64]);  mul_123 = None
        permute_170: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_121: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_371: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [512, 128, 64]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_364: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_451, [4096, 1024])
        permute_166: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_91: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg248_1, view_364, permute_166);  arg248_1 = view_364 = permute_166 = None
        view_365: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_91, [32, 128, 1024]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_368: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [32, 128, -1, 64]);  view_365 = None
        permute_168: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_122: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_372: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [512, 128, 64]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_171: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None
        bmm_30: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_371, permute_171);  view_371 = permute_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_374: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [32, 16, 128, 128]);  bmm_30 = None
        add_126: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_374, where);  view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_17: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_15: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_126, full_default_17);  add_126 = full_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_375: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_15, [512, 128, 128]);  maximum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_463: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_375, torch.float32);  view_375 = None
        amax_15: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_463, [-1], True)
        sub_46: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_463, amax_15);  convert_element_type_463 = amax_15 = None
        exp_15: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_464: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_366: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_451, [4096, 1024]);  convert_element_type_451 = None
        permute_167: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_92: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg250_1, view_366, permute_167);  arg250_1 = view_366 = permute_167 = None
        view_367: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_92, [32, 128, 1024]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_369: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_367, [32, 128, -1, 64]);  view_367 = None
        permute_169: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_123: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_373: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [512, 128, 64]);  clone_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_31: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_464, view_373);  convert_element_type_464 = view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_378: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [32, 16, 128, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_172: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_125: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_379: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_125, [32, 128, 1024]);  clone_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_380: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_379, [4096, 1024]);  view_379 = None
        permute_173: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_93: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg252_1, view_380, permute_173);  arg252_1 = view_380 = permute_173 = None
        view_381: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_93, [32, 128, 1024]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_127: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, view_381);  add_123 = view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_470: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_470, [2], correction = 0, keepdim = True)
        getitem_62: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        sub_47: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_470, getitem_63);  convert_element_type_470 = getitem_63 = None
        add_128: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_124: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_125: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, arg253_1);  mul_124 = arg253_1 = None
        add_129: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, arg254_1);  mul_125 = arg254_1 = None
        convert_element_type_471: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_382: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_471, [4096, 1024]);  convert_element_type_471 = None
        permute_174: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_94: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg256_1, view_382, permute_174);  arg256_1 = view_382 = permute_174 = None
        view_383: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [32, 128, 4096]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_475: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_383, torch.float32);  view_383 = None
        mul_126: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, 0.5)
        mul_127: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, 0.7071067811865476);  convert_element_type_475 = None
        erf_15: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_127);  mul_127 = None
        add_130: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_128: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, add_130);  mul_126 = add_130 = None
        convert_element_type_476: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_128, torch.bfloat16);  mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_384: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_476, [4096, 4096]);  convert_element_type_476 = None
        permute_175: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_95: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg258_1, view_384, permute_175);  arg258_1 = view_384 = permute_175 = None
        view_385: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_95, [32, 128, 1024]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_131: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_127, view_385);  add_127 = view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_480: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_480, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant17: "f32[][]cuda:0" = self._tensor_constant17;  _tensor_constant17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_48: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_480, getitem_65);  convert_element_type_480 = getitem_65 = None
        add_132: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_129: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
        mul_130: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, arg259_1);  mul_129 = arg259_1 = None
        add_133: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, arg260_1);  mul_130 = arg260_1 = None
        convert_element_type_481: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_386: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [4096, 1024])
        permute_176: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_96: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg262_1, view_386, permute_176);  arg262_1 = view_386 = permute_176 = None
        view_387: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_96, [32, 128, 1024]);  addmm_96 = None
        mul_131: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_387, 0.125);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_394: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_131, [32, 128, 16, 64]);  mul_131 = None
        permute_181: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_129: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_395: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [512, 128, 64]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_388: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [4096, 1024])
        permute_177: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_97: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg264_1, view_388, permute_177);  arg264_1 = view_388 = permute_177 = None
        view_389: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_97, [32, 128, 1024]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_392: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_389, [32, 128, -1, 64]);  view_389 = None
        permute_179: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_130: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_396: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [512, 128, 64]);  clone_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_182: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_396, [0, 2, 1]);  view_396 = None
        bmm_32: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_395, permute_182);  view_395 = permute_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_398: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [32, 16, 128, 128]);  bmm_32 = None
        add_134: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_398, where);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_18: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_16: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_134, full_default_18);  add_134 = full_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_399: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_16, [512, 128, 128]);  maximum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_493: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_399, torch.float32);  view_399 = None
        amax_16: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_493, [-1], True)
        sub_49: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_493, amax_16);  convert_element_type_493 = amax_16 = None
        exp_16: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_494: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_390: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [4096, 1024]);  convert_element_type_481 = None
        permute_178: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_98: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg266_1, view_390, permute_178);  arg266_1 = view_390 = permute_178 = None
        view_391: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_98, [32, 128, 1024]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_393: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_391, [32, 128, -1, 64]);  view_391 = None
        permute_180: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_131: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_397: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [512, 128, 64]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_33: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_494, view_397);  convert_element_type_494 = view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_402: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [32, 16, 128, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_183: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_133: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_403: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_133, [32, 128, 1024]);  clone_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_404: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_403, [4096, 1024]);  view_403 = None
        permute_184: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_99: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg268_1, view_404, permute_184);  arg268_1 = view_404 = permute_184 = None
        view_405: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_99, [32, 128, 1024]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_135: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_131, view_405);  add_131 = view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_500: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_500, [2], correction = 0, keepdim = True)
        getitem_66: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        sub_50: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_500, getitem_67);  convert_element_type_500 = getitem_67 = None
        add_136: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_132: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
        mul_133: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, arg269_1);  mul_132 = arg269_1 = None
        add_137: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_133, arg270_1);  mul_133 = arg270_1 = None
        convert_element_type_501: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_406: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_501, [4096, 1024]);  convert_element_type_501 = None
        permute_185: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_100: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg272_1, view_406, permute_185);  arg272_1 = view_406 = permute_185 = None
        view_407: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_100, [32, 128, 4096]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_505: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        mul_134: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_505, 0.5)
        mul_135: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_505, 0.7071067811865476);  convert_element_type_505 = None
        erf_16: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
        add_138: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_136: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, add_138);  mul_134 = add_138 = None
        convert_element_type_506: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_408: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_506, [4096, 4096]);  convert_element_type_506 = None
        permute_186: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_101: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg274_1, view_408, permute_186);  arg274_1 = view_408 = permute_186 = None
        view_409: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_101, [32, 128, 1024]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_139: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, view_409);  add_135 = view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_510: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_510, [2], correction = 0, keepdim = True)
        getitem_68: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant18: "f32[][]cuda:0" = self._tensor_constant18;  _tensor_constant18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_51: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_510, getitem_69);  convert_element_type_510 = getitem_69 = None
        add_140: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_137: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
        mul_138: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, arg275_1);  mul_137 = arg275_1 = None
        add_141: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_138, arg276_1);  mul_138 = arg276_1 = None
        convert_element_type_511: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_410: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_511, [4096, 1024])
        permute_187: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_102: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg278_1, view_410, permute_187);  arg278_1 = view_410 = permute_187 = None
        view_411: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_102, [32, 128, 1024]);  addmm_102 = None
        mul_139: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_411, 0.125);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_418: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_139, [32, 128, 16, 64]);  mul_139 = None
        permute_192: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_137: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_419: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [512, 128, 64]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_412: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_511, [4096, 1024])
        permute_188: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_103: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg280_1, view_412, permute_188);  arg280_1 = view_412 = permute_188 = None
        view_413: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_103, [32, 128, 1024]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_416: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_413, [32, 128, -1, 64]);  view_413 = None
        permute_190: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_138: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_420: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [512, 128, 64]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_193: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 2, 1]);  view_420 = None
        bmm_34: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_419, permute_193);  view_419 = permute_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_422: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [32, 16, 128, 128]);  bmm_34 = None
        add_142: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_422, where);  view_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_19: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_17: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_142, full_default_19);  add_142 = full_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_423: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_17, [512, 128, 128]);  maximum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_523: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_423, torch.float32);  view_423 = None
        amax_17: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_523, [-1], True)
        sub_52: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_523, amax_17);  convert_element_type_523 = amax_17 = None
        exp_17: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_524: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_414: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_511, [4096, 1024]);  convert_element_type_511 = None
        permute_189: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_104: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg282_1, view_414, permute_189);  arg282_1 = view_414 = permute_189 = None
        view_415: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_104, [32, 128, 1024]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_417: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_415, [32, 128, -1, 64]);  view_415 = None
        permute_191: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_139: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
        view_421: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_139, [512, 128, 64]);  clone_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_35: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_524, view_421);  convert_element_type_524 = view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_426: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [32, 16, 128, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_194: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_141: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_427: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_141, [32, 128, 1024]);  clone_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_428: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_427, [4096, 1024]);  view_427 = None
        permute_195: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_105: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg284_1, view_428, permute_195);  arg284_1 = view_428 = permute_195 = None
        view_429: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_105, [32, 128, 1024]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_143: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_139, view_429);  add_139 = view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_530: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_530, [2], correction = 0, keepdim = True)
        getitem_70: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        sub_53: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_530, getitem_71);  convert_element_type_530 = getitem_71 = None
        add_144: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_140: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
        mul_141: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, arg285_1);  mul_140 = arg285_1 = None
        add_145: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_141, arg286_1);  mul_141 = arg286_1 = None
        convert_element_type_531: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_430: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_531, [4096, 1024]);  convert_element_type_531 = None
        permute_196: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_106: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg288_1, view_430, permute_196);  arg288_1 = view_430 = permute_196 = None
        view_431: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_106, [32, 128, 4096]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_535: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_431, torch.float32);  view_431 = None
        mul_142: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_535, 0.5)
        mul_143: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_535, 0.7071067811865476);  convert_element_type_535 = None
        erf_17: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
        add_146: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_144: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_142, add_146);  mul_142 = add_146 = None
        convert_element_type_536: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_432: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_536, [4096, 4096]);  convert_element_type_536 = None
        permute_197: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_107: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg290_1, view_432, permute_197);  arg290_1 = view_432 = permute_197 = None
        view_433: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_107, [32, 128, 1024]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_147: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_143, view_433);  add_143 = view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_540: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_540, [2], correction = 0, keepdim = True)
        getitem_72: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant19: "f32[][]cuda:0" = self._tensor_constant19;  _tensor_constant19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_54: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_540, getitem_73);  convert_element_type_540 = getitem_73 = None
        add_148: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_145: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
        mul_146: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, arg291_1);  mul_145 = arg291_1 = None
        add_149: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, arg292_1);  mul_146 = arg292_1 = None
        convert_element_type_541: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_434: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_541, [4096, 1024])
        permute_198: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_108: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg294_1, view_434, permute_198);  arg294_1 = view_434 = permute_198 = None
        view_435: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_108, [32, 128, 1024]);  addmm_108 = None
        mul_147: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_435, 0.125);  view_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_442: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_147, [32, 128, 16, 64]);  mul_147 = None
        permute_203: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_145: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_443: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [512, 128, 64]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_436: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_541, [4096, 1024])
        permute_199: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_109: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg296_1, view_436, permute_199);  arg296_1 = view_436 = permute_199 = None
        view_437: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_109, [32, 128, 1024]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_440: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_437, [32, 128, -1, 64]);  view_437 = None
        permute_201: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_146: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_444: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [512, 128, 64]);  clone_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_204: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
        bmm_36: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_443, permute_204);  view_443 = permute_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_446: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [32, 16, 128, 128]);  bmm_36 = None
        add_150: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_446, where);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_20: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_18: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_150, full_default_20);  add_150 = full_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_447: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_18, [512, 128, 128]);  maximum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_553: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_447, torch.float32);  view_447 = None
        amax_18: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_553, [-1], True)
        sub_55: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_553, amax_18);  convert_element_type_553 = amax_18 = None
        exp_18: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_554: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_438: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_541, [4096, 1024]);  convert_element_type_541 = None
        permute_200: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_110: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg298_1, view_438, permute_200);  arg298_1 = view_438 = permute_200 = None
        view_439: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_110, [32, 128, 1024]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_441: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_439, [32, 128, -1, 64]);  view_439 = None
        permute_202: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_147: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_445: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_147, [512, 128, 64]);  clone_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_37: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_554, view_445);  convert_element_type_554 = view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_450: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [32, 16, 128, 64]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_205: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_149: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_451: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_149, [32, 128, 1024]);  clone_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_452: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_451, [4096, 1024]);  view_451 = None
        permute_206: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_111: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg300_1, view_452, permute_206);  arg300_1 = view_452 = permute_206 = None
        view_453: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_111, [32, 128, 1024]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_151: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_147, view_453);  add_147 = view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_560: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_560, [2], correction = 0, keepdim = True)
        getitem_74: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        sub_56: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_560, getitem_75);  convert_element_type_560 = getitem_75 = None
        add_152: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_148: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
        mul_149: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, arg301_1);  mul_148 = arg301_1 = None
        add_153: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, arg302_1);  mul_149 = arg302_1 = None
        convert_element_type_561: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_454: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_561, [4096, 1024]);  convert_element_type_561 = None
        permute_207: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_112: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg304_1, view_454, permute_207);  arg304_1 = view_454 = permute_207 = None
        view_455: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_112, [32, 128, 4096]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_565: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_455, torch.float32);  view_455 = None
        mul_150: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_565, 0.5)
        mul_151: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_565, 0.7071067811865476);  convert_element_type_565 = None
        erf_18: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_154: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_152: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, add_154);  mul_150 = add_154 = None
        convert_element_type_566: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_456: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_566, [4096, 4096]);  convert_element_type_566 = None
        permute_208: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_113: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg306_1, view_456, permute_208);  arg306_1 = view_456 = permute_208 = None
        view_457: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_113, [32, 128, 1024]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_155: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_151, view_457);  add_151 = view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_570: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_155, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_570, [2], correction = 0, keepdim = True)
        getitem_76: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant20: "f32[][]cuda:0" = self._tensor_constant20;  _tensor_constant20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_57: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_570, getitem_77);  convert_element_type_570 = getitem_77 = None
        add_156: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        mul_153: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
        mul_154: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_153, arg307_1);  mul_153 = arg307_1 = None
        add_157: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_154, arg308_1);  mul_154 = arg308_1 = None
        convert_element_type_571: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_458: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_571, [4096, 1024])
        permute_209: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_114: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg310_1, view_458, permute_209);  arg310_1 = view_458 = permute_209 = None
        view_459: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_114, [32, 128, 1024]);  addmm_114 = None
        mul_155: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_459, 0.125);  view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_466: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_155, [32, 128, 16, 64]);  mul_155 = None
        permute_214: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_153: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_467: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [512, 128, 64]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_460: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_571, [4096, 1024])
        permute_210: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_115: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg312_1, view_460, permute_210);  arg312_1 = view_460 = permute_210 = None
        view_461: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_115, [32, 128, 1024]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_464: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_461, [32, 128, -1, 64]);  view_461 = None
        permute_212: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_154: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_468: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [512, 128, 64]);  clone_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_215: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_468, [0, 2, 1]);  view_468 = None
        bmm_38: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_467, permute_215);  view_467 = permute_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_470: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [32, 16, 128, 128]);  bmm_38 = None
        add_158: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_470, where);  view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_21: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_19: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_158, full_default_21);  add_158 = full_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_471: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_19, [512, 128, 128]);  maximum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_583: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_471, torch.float32);  view_471 = None
        amax_19: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_583, [-1], True)
        sub_58: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_583, amax_19);  convert_element_type_583 = amax_19 = None
        exp_19: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_584: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_462: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_571, [4096, 1024]);  convert_element_type_571 = None
        permute_211: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_116: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg314_1, view_462, permute_211);  arg314_1 = view_462 = permute_211 = None
        view_463: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_116, [32, 128, 1024]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_465: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_463, [32, 128, -1, 64]);  view_463 = None
        permute_213: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_155: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_469: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_155, [512, 128, 64]);  clone_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_39: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_584, view_469);  convert_element_type_584 = view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_474: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [32, 16, 128, 64]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_216: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_157: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_475: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_157, [32, 128, 1024]);  clone_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_476: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_475, [4096, 1024]);  view_475 = None
        permute_217: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_117: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg316_1, view_476, permute_217);  arg316_1 = view_476 = permute_217 = None
        view_477: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_117, [32, 128, 1024]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_159: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_155, view_477);  add_155 = view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_590: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_590, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        sub_59: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_590, getitem_79);  convert_element_type_590 = getitem_79 = None
        add_160: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_156: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
        mul_157: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, arg317_1);  mul_156 = arg317_1 = None
        add_161: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_157, arg318_1);  mul_157 = arg318_1 = None
        convert_element_type_591: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.bfloat16);  add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_478: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_591, [4096, 1024]);  convert_element_type_591 = None
        permute_218: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_118: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg320_1, view_478, permute_218);  arg320_1 = view_478 = permute_218 = None
        view_479: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_118, [32, 128, 4096]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_595: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_479, torch.float32);  view_479 = None
        mul_158: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_595, 0.5)
        mul_159: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_595, 0.7071067811865476);  convert_element_type_595 = None
        erf_19: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
        add_162: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_160: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, add_162);  mul_158 = add_162 = None
        convert_element_type_596: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_160, torch.bfloat16);  mul_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_480: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_596, [4096, 4096]);  convert_element_type_596 = None
        permute_219: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_119: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg322_1, view_480, permute_219);  arg322_1 = view_480 = permute_219 = None
        view_481: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_119, [32, 128, 1024]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_163: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_159, view_481);  add_159 = view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_600: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_163, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_600, [2], correction = 0, keepdim = True)
        getitem_80: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant21: "f32[][]cuda:0" = self._tensor_constant21;  _tensor_constant21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_60: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_600, getitem_81);  convert_element_type_600 = getitem_81 = None
        add_164: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        mul_161: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
        mul_162: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, arg323_1);  mul_161 = arg323_1 = None
        add_165: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, arg324_1);  mul_162 = arg324_1 = None
        convert_element_type_601: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_482: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_601, [4096, 1024])
        permute_220: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_120: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg326_1, view_482, permute_220);  arg326_1 = view_482 = permute_220 = None
        view_483: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_120, [32, 128, 1024]);  addmm_120 = None
        mul_163: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_483, 0.125);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_490: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_163, [32, 128, 16, 64]);  mul_163 = None
        permute_225: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_161: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_491: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [512, 128, 64]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_484: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_601, [4096, 1024])
        permute_221: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_121: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg328_1, view_484, permute_221);  arg328_1 = view_484 = permute_221 = None
        view_485: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_121, [32, 128, 1024]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_488: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [32, 128, -1, 64]);  view_485 = None
        permute_223: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_162: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_492: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [512, 128, 64]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_226: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_492, [0, 2, 1]);  view_492 = None
        bmm_40: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_491, permute_226);  view_491 = permute_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_494: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [32, 16, 128, 128]);  bmm_40 = None
        add_166: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_494, where);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_22: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_20: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_166, full_default_22);  add_166 = full_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_495: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_20, [512, 128, 128]);  maximum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_613: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_495, torch.float32);  view_495 = None
        amax_20: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_613, [-1], True)
        sub_61: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_613, amax_20);  convert_element_type_613 = amax_20 = None
        exp_20: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_614: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_486: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_601, [4096, 1024]);  convert_element_type_601 = None
        permute_222: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_122: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg330_1, view_486, permute_222);  arg330_1 = view_486 = permute_222 = None
        view_487: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_122, [32, 128, 1024]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_489: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_487, [32, 128, -1, 64]);  view_487 = None
        permute_224: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_163: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_493: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_163, [512, 128, 64]);  clone_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_41: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_614, view_493);  convert_element_type_614 = view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_498: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [32, 16, 128, 64]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_227: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_165: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_499: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_165, [32, 128, 1024]);  clone_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_500: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_499, [4096, 1024]);  view_499 = None
        permute_228: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_123: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg332_1, view_500, permute_228);  arg332_1 = view_500 = permute_228 = None
        view_501: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_123, [32, 128, 1024]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_167: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_163, view_501);  add_163 = view_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_620: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_620, [2], correction = 0, keepdim = True)
        getitem_82: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        sub_62: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_620, getitem_83);  convert_element_type_620 = getitem_83 = None
        add_168: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_164: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
        mul_165: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, arg333_1);  mul_164 = arg333_1 = None
        add_169: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, arg334_1);  mul_165 = arg334_1 = None
        convert_element_type_621: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_502: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_621, [4096, 1024]);  convert_element_type_621 = None
        permute_229: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_124: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg336_1, view_502, permute_229);  arg336_1 = view_502 = permute_229 = None
        view_503: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_124, [32, 128, 4096]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_625: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        mul_166: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_625, 0.5)
        mul_167: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_625, 0.7071067811865476);  convert_element_type_625 = None
        erf_20: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_170: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_168: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, add_170);  mul_166 = add_170 = None
        convert_element_type_626: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_504: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_626, [4096, 4096]);  convert_element_type_626 = None
        permute_230: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_125: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg338_1, view_504, permute_230);  arg338_1 = view_504 = permute_230 = None
        view_505: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_125, [32, 128, 1024]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_171: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_167, view_505);  add_167 = view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_630: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_630, [2], correction = 0, keepdim = True)
        getitem_84: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant22: "f32[][]cuda:0" = self._tensor_constant22;  _tensor_constant22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_63: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_630, getitem_85);  convert_element_type_630 = getitem_85 = None
        add_172: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        mul_169: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
        mul_170: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, arg339_1);  mul_169 = arg339_1 = None
        add_173: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, arg340_1);  mul_170 = arg340_1 = None
        convert_element_type_631: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_506: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_631, [4096, 1024])
        permute_231: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_126: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg342_1, view_506, permute_231);  arg342_1 = view_506 = permute_231 = None
        view_507: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_126, [32, 128, 1024]);  addmm_126 = None
        mul_171: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_507, 0.125);  view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_514: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_171, [32, 128, 16, 64]);  mul_171 = None
        permute_236: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_169: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_515: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [512, 128, 64]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_508: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_631, [4096, 1024])
        permute_232: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_127: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg344_1, view_508, permute_232);  arg344_1 = view_508 = permute_232 = None
        view_509: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_127, [32, 128, 1024]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_512: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_509, [32, 128, -1, 64]);  view_509 = None
        permute_234: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_170: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_516: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [512, 128, 64]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_237: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_516, [0, 2, 1]);  view_516 = None
        bmm_42: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_515, permute_237);  view_515 = permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_518: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [32, 16, 128, 128]);  bmm_42 = None
        add_174: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_518, where);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_23: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_21: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_174, full_default_23);  add_174 = full_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_519: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_21, [512, 128, 128]);  maximum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_643: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_519, torch.float32);  view_519 = None
        amax_21: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_643, [-1], True)
        sub_64: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_643, amax_21);  convert_element_type_643 = amax_21 = None
        exp_21: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_644: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_510: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_631, [4096, 1024]);  convert_element_type_631 = None
        permute_233: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_128: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg346_1, view_510, permute_233);  arg346_1 = view_510 = permute_233 = None
        view_511: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_128, [32, 128, 1024]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_513: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [32, 128, -1, 64]);  view_511 = None
        permute_235: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_171: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
        view_517: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [512, 128, 64]);  clone_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_43: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_644, view_517);  convert_element_type_644 = view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_522: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [32, 16, 128, 64]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_238: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_173: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_523: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_173, [32, 128, 1024]);  clone_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_524: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_523, [4096, 1024]);  view_523 = None
        permute_239: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_129: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg348_1, view_524, permute_239);  arg348_1 = view_524 = permute_239 = None
        view_525: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_129, [32, 128, 1024]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_175: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_171, view_525);  add_171 = view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_650: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_650, [2], correction = 0, keepdim = True)
        getitem_86: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        sub_65: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_650, getitem_87);  convert_element_type_650 = getitem_87 = None
        add_176: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        mul_172: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
        mul_173: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, arg349_1);  mul_172 = arg349_1 = None
        add_177: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_173, arg350_1);  mul_173 = arg350_1 = None
        convert_element_type_651: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.bfloat16);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_526: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_651, [4096, 1024]);  convert_element_type_651 = None
        permute_240: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_130: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg352_1, view_526, permute_240);  arg352_1 = view_526 = permute_240 = None
        view_527: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_130, [32, 128, 4096]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_655: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_527, torch.float32);  view_527 = None
        mul_174: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_655, 0.5)
        mul_175: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_655, 0.7071067811865476);  convert_element_type_655 = None
        erf_21: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
        add_178: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_176: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, add_178);  mul_174 = add_178 = None
        convert_element_type_656: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_176, torch.bfloat16);  mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_528: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_656, [4096, 4096]);  convert_element_type_656 = None
        permute_241: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_131: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg354_1, view_528, permute_241);  arg354_1 = view_528 = permute_241 = None
        view_529: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_131, [32, 128, 1024]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_179: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_175, view_529);  add_175 = view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_660: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_660, [2], correction = 0, keepdim = True)
        getitem_88: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant23: "f32[][]cuda:0" = self._tensor_constant23;  _tensor_constant23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_66: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_660, getitem_89);  convert_element_type_660 = getitem_89 = None
        add_180: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        mul_177: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
        mul_178: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, arg355_1);  mul_177 = arg355_1 = None
        add_181: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_178, arg356_1);  mul_178 = arg356_1 = None
        convert_element_type_661: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_181, torch.bfloat16);  add_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_530: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_661, [4096, 1024])
        permute_242: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_132: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg358_1, view_530, permute_242);  arg358_1 = view_530 = permute_242 = None
        view_531: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_132, [32, 128, 1024]);  addmm_132 = None
        mul_179: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_531, 0.125);  view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_538: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_179, [32, 128, 16, 64]);  mul_179 = None
        permute_247: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_177: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_539: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [512, 128, 64]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_532: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_661, [4096, 1024])
        permute_243: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_133: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg360_1, view_532, permute_243);  arg360_1 = view_532 = permute_243 = None
        view_533: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_133, [32, 128, 1024]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_536: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_533, [32, 128, -1, 64]);  view_533 = None
        permute_245: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_536, [0, 2, 1, 3]);  view_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_178: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
        view_540: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [512, 128, 64]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_248: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_540, [0, 2, 1]);  view_540 = None
        bmm_44: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_539, permute_248);  view_539 = permute_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_542: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [32, 16, 128, 128]);  bmm_44 = None
        add_182: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_542, where);  view_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_24: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_22: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_182, full_default_24);  add_182 = full_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_543: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_22, [512, 128, 128]);  maximum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_673: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_543, torch.float32);  view_543 = None
        amax_22: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_673, [-1], True)
        sub_67: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_673, amax_22);  convert_element_type_673 = amax_22 = None
        exp_22: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_674: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_534: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_661, [4096, 1024]);  convert_element_type_661 = None
        permute_244: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_134: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg362_1, view_534, permute_244);  arg362_1 = view_534 = permute_244 = None
        view_535: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_134, [32, 128, 1024]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_537: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_535, [32, 128, -1, 64]);  view_535 = None
        permute_246: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_179: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_541: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [512, 128, 64]);  clone_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_45: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_674, view_541);  convert_element_type_674 = view_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_546: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [32, 16, 128, 64]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_249: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_181: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_547: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_181, [32, 128, 1024]);  clone_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_548: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_547, [4096, 1024]);  view_547 = None
        permute_250: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_135: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg364_1, view_548, permute_250);  arg364_1 = view_548 = permute_250 = None
        view_549: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_135, [32, 128, 1024]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_183: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_179, view_549);  add_179 = view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_680: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_183, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_680, [2], correction = 0, keepdim = True)
        getitem_90: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        sub_68: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_680, getitem_91);  convert_element_type_680 = getitem_91 = None
        add_184: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        mul_180: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
        mul_181: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, arg365_1);  mul_180 = arg365_1 = None
        add_185: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_181, arg366_1);  mul_181 = arg366_1 = None
        convert_element_type_681: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_185, torch.bfloat16);  add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_550: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_681, [4096, 1024]);  convert_element_type_681 = None
        permute_251: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_136: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg368_1, view_550, permute_251);  arg368_1 = view_550 = permute_251 = None
        view_551: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_136, [32, 128, 4096]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_685: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_551, torch.float32);  view_551 = None
        mul_182: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_685, 0.5)
        mul_183: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_685, 0.7071067811865476);  convert_element_type_685 = None
        erf_22: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_183);  mul_183 = None
        add_186: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_184: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, add_186);  mul_182 = add_186 = None
        convert_element_type_686: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_184, torch.bfloat16);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_552: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_686, [4096, 4096]);  convert_element_type_686 = None
        permute_252: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_137: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg370_1, view_552, permute_252);  arg370_1 = view_552 = permute_252 = None
        view_553: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_137, [32, 128, 1024]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_187: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_183, view_553);  add_183 = view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_690: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_187, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_690, [2], correction = 0, keepdim = True)
        getitem_92: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_46[0]
        getitem_93: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        _tensor_constant24: "f32[][]cuda:0" = self._tensor_constant24;  _tensor_constant24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_69: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_690, getitem_93);  convert_element_type_690 = getitem_93 = None
        add_188: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_185: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
        mul_186: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, arg371_1);  mul_185 = arg371_1 = None
        add_189: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_186, arg372_1);  mul_186 = arg372_1 = None
        convert_element_type_691: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_554: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_691, [4096, 1024])
        permute_253: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_138: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg374_1, view_554, permute_253);  arg374_1 = view_554 = permute_253 = None
        view_555: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_138, [32, 128, 1024]);  addmm_138 = None
        mul_187: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_555, 0.125);  view_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_562: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_187, [32, 128, 16, 64]);  mul_187 = None
        permute_258: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_185: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_563: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [512, 128, 64]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        view_556: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_691, [4096, 1024])
        permute_254: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        addmm_139: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg376_1, view_556, permute_254);  arg376_1 = view_556 = permute_254 = None
        view_557: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_139, [32, 128, 1024]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_560: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_557, [32, 128, -1, 64]);  view_557 = None
        permute_256: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_186: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_564: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [512, 128, 64]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_259: "bf16[512, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_564, [0, 2, 1]);  view_564 = None
        bmm_46: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_563, permute_259);  view_563 = permute_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_566: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [32, 16, 128, 128]);  bmm_46 = None
        add_190: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_566, where);  view_566 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_25: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_23: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_190, full_default_25);  add_190 = full_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_567: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum_23, [512, 128, 128]);  maximum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_703: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_567, torch.float32);  view_567 = None
        amax_23: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_703, [-1], True)
        sub_70: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_703, amax_23);  convert_element_type_703 = amax_23 = None
        exp_23: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_704: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_558: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_691, [4096, 1024]);  convert_element_type_691 = None
        permute_255: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_140: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg378_1, view_558, permute_255);  arg378_1 = view_558 = permute_255 = None
        view_559: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_140, [32, 128, 1024]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        view_561: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_559, [32, 128, -1, 64]);  view_559 = None
        permute_257: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_187: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
        view_565: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_187, [512, 128, 64]);  clone_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_47: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_704, view_565);  convert_element_type_704 = view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_570: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [32, 16, 128, 64]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_260: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_189: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_571: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [32, 128, 1024]);  clone_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_572: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_571, [4096, 1024]);  view_571 = None
        permute_261: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_141: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg380_1, view_572, permute_261);  arg380_1 = view_572 = permute_261 = None
        view_573: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_141, [32, 128, 1024]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_191: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_187, view_573);  add_187 = view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_710: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_710, [2], correction = 0, keepdim = True)
        getitem_94: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_47[0]
        getitem_95: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        sub_71: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_710, getitem_95);  convert_element_type_710 = getitem_95 = None
        add_192: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_188: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
        mul_189: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_188, arg381_1);  mul_188 = arg381_1 = None
        add_193: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_189, arg382_1);  mul_189 = arg382_1 = None
        convert_element_type_711: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16);  add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_574: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_711, [4096, 1024]);  convert_element_type_711 = None
        permute_262: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_142: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg384_1, view_574, permute_262);  arg384_1 = view_574 = permute_262 = None
        view_575: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_142, [32, 128, 4096]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_715: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_575, torch.float32);  view_575 = None
        mul_190: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_715, 0.5)
        mul_191: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_715, 0.7071067811865476);  convert_element_type_715 = None
        erf_23: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
        add_194: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_192: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, add_194);  mul_190 = add_194 = None
        convert_element_type_716: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_576: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_716, [4096, 4096]);  convert_element_type_716 = None
        permute_263: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_143: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg386_1, view_576, permute_263);  arg386_1 = view_576 = permute_263 = None
        view_577: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_143, [32, 128, 1024]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_195: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_191, view_577);  add_191 = view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:480 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_720: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_195, torch.float32);  add_195 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_720, [2], correction = 0, keepdim = True)
        getitem_96: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_48[0]
        getitem_97: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[32, 129][129, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(arg389_1, [0, 1], -100.0);  arg389_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[32, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_193: "i64[32, 128][128, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_581: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [-1]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_1: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_581, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:480 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_72: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_720, getitem_97);  convert_element_type_720 = getitem_97 = None
        add_196: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        mul_193: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
        mul_194: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_193, arg387_1);  mul_193 = arg387_1 = None
        add_197: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_194, arg388_1);  mul_194 = arg388_1 = None
        convert_element_type_721: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16);  add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:556 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_578: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_721, [4096, 1024]);  convert_element_type_721 = None
        permute_264: "bf16[1024, 256008][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "bf16[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.mm.default(view_578, permute_264);  view_578 = permute_264 = None
        view_579: "bf16[32, 128, 256008][32769024, 256008, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 128, 256008]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_724: "f32[32, 128, 256008][32769024, 256008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_579, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_580: "f32[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_724, [-1, 256008]);  convert_element_type_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_24: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(view_580, [1], True)
        sub_73: "f32[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_580, amax_24);  view_580 = amax_24 = None
        exp_24: "f32[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.exp.default(sub_73)
        sum_25: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_74: "f32[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_73, log);  sub_73 = log = None
        ne: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_581, -100)
        full_default_26: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[4096][1]cuda:0" = torch.ops.aten.where.self(ne, view_581, full_default_26);  ne = full_default_26 = None
        unsqueeze_7: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(sub_74, 1, unsqueeze_7);  sub_74 = unsqueeze_7 = None
        squeeze: "f32[4096][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_27: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4096][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_27);  ne_1 = neg = full_default_27 = None
        sum_27: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_2);  where_2 = None
        ne_2: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_581, -100);  view_581 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_725: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        div_24: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_725);  sum_27 = convert_element_type_725 = None
        return (div_24, view_579)

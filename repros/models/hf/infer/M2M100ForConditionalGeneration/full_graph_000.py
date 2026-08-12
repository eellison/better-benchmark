class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[64, 128][128, 1]cuda:0", arg1_1: "i64[64, 128][128, 1]cuda:0", arg2_1: "bf16[128112, 1024][1024, 1]cuda:0", arg3_1: "bf16[1026, 1024][1024, 1]cuda:0", arg4_1: "bf16[1024][1]cuda:0", arg5_1: "bf16[1024][1]cuda:0", arg6_1: "bf16[1024, 1024][1024, 1]cuda:0", arg7_1: "bf16[1024][1]cuda:0", arg8_1: "bf16[1024, 1024][1024, 1]cuda:0", arg9_1: "bf16[1024][1]cuda:0", arg10_1: "bf16[1024, 1024][1024, 1]cuda:0", arg11_1: "bf16[1024][1]cuda:0", arg12_1: "bf16[1024, 1024][1024, 1]cuda:0", arg13_1: "bf16[1024][1]cuda:0", arg14_1: "bf16[1024][1]cuda:0", arg15_1: "bf16[1024][1]cuda:0", arg16_1: "bf16[4096, 1024][1024, 1]cuda:0", arg17_1: "bf16[4096][1]cuda:0", arg18_1: "bf16[1024, 4096][4096, 1]cuda:0", arg19_1: "bf16[1024][1]cuda:0", arg20_1: "bf16[1024][1]cuda:0", arg21_1: "bf16[1024][1]cuda:0", arg22_1: "bf16[1024, 1024][1024, 1]cuda:0", arg23_1: "bf16[1024][1]cuda:0", arg24_1: "bf16[1024, 1024][1024, 1]cuda:0", arg25_1: "bf16[1024][1]cuda:0", arg26_1: "bf16[1024, 1024][1024, 1]cuda:0", arg27_1: "bf16[1024][1]cuda:0", arg28_1: "bf16[1024, 1024][1024, 1]cuda:0", arg29_1: "bf16[1024][1]cuda:0", arg30_1: "bf16[1024][1]cuda:0", arg31_1: "bf16[1024][1]cuda:0", arg32_1: "bf16[4096, 1024][1024, 1]cuda:0", arg33_1: "bf16[4096][1]cuda:0", arg34_1: "bf16[1024, 4096][4096, 1]cuda:0", arg35_1: "bf16[1024][1]cuda:0", arg36_1: "bf16[1024][1]cuda:0", arg37_1: "bf16[1024][1]cuda:0", arg38_1: "bf16[1024, 1024][1024, 1]cuda:0", arg39_1: "bf16[1024][1]cuda:0", arg40_1: "bf16[1024, 1024][1024, 1]cuda:0", arg41_1: "bf16[1024][1]cuda:0", arg42_1: "bf16[1024, 1024][1024, 1]cuda:0", arg43_1: "bf16[1024][1]cuda:0", arg44_1: "bf16[1024, 1024][1024, 1]cuda:0", arg45_1: "bf16[1024][1]cuda:0", arg46_1: "bf16[1024][1]cuda:0", arg47_1: "bf16[1024][1]cuda:0", arg48_1: "bf16[4096, 1024][1024, 1]cuda:0", arg49_1: "bf16[4096][1]cuda:0", arg50_1: "bf16[1024, 4096][4096, 1]cuda:0", arg51_1: "bf16[1024][1]cuda:0", arg52_1: "bf16[1024][1]cuda:0", arg53_1: "bf16[1024][1]cuda:0", arg54_1: "bf16[1024, 1024][1024, 1]cuda:0", arg55_1: "bf16[1024][1]cuda:0", arg56_1: "bf16[1024, 1024][1024, 1]cuda:0", arg57_1: "bf16[1024][1]cuda:0", arg58_1: "bf16[1024, 1024][1024, 1]cuda:0", arg59_1: "bf16[1024][1]cuda:0", arg60_1: "bf16[1024, 1024][1024, 1]cuda:0", arg61_1: "bf16[1024][1]cuda:0", arg62_1: "bf16[1024][1]cuda:0", arg63_1: "bf16[1024][1]cuda:0", arg64_1: "bf16[4096, 1024][1024, 1]cuda:0", arg65_1: "bf16[4096][1]cuda:0", arg66_1: "bf16[1024, 4096][4096, 1]cuda:0", arg67_1: "bf16[1024][1]cuda:0", arg68_1: "bf16[1024][1]cuda:0", arg69_1: "bf16[1024][1]cuda:0", arg70_1: "bf16[1024, 1024][1024, 1]cuda:0", arg71_1: "bf16[1024][1]cuda:0", arg72_1: "bf16[1024, 1024][1024, 1]cuda:0", arg73_1: "bf16[1024][1]cuda:0", arg74_1: "bf16[1024, 1024][1024, 1]cuda:0", arg75_1: "bf16[1024][1]cuda:0", arg76_1: "bf16[1024, 1024][1024, 1]cuda:0", arg77_1: "bf16[1024][1]cuda:0", arg78_1: "bf16[1024][1]cuda:0", arg79_1: "bf16[1024][1]cuda:0", arg80_1: "bf16[4096, 1024][1024, 1]cuda:0", arg81_1: "bf16[4096][1]cuda:0", arg82_1: "bf16[1024, 4096][4096, 1]cuda:0", arg83_1: "bf16[1024][1]cuda:0", arg84_1: "bf16[1024][1]cuda:0", arg85_1: "bf16[1024][1]cuda:0", arg86_1: "bf16[1024, 1024][1024, 1]cuda:0", arg87_1: "bf16[1024][1]cuda:0", arg88_1: "bf16[1024, 1024][1024, 1]cuda:0", arg89_1: "bf16[1024][1]cuda:0", arg90_1: "bf16[1024, 1024][1024, 1]cuda:0", arg91_1: "bf16[1024][1]cuda:0", arg92_1: "bf16[1024, 1024][1024, 1]cuda:0", arg93_1: "bf16[1024][1]cuda:0", arg94_1: "bf16[1024][1]cuda:0", arg95_1: "bf16[1024][1]cuda:0", arg96_1: "bf16[4096, 1024][1024, 1]cuda:0", arg97_1: "bf16[4096][1]cuda:0", arg98_1: "bf16[1024, 4096][4096, 1]cuda:0", arg99_1: "bf16[1024][1]cuda:0", arg100_1: "bf16[1024][1]cuda:0", arg101_1: "bf16[1024][1]cuda:0", arg102_1: "bf16[1024, 1024][1024, 1]cuda:0", arg103_1: "bf16[1024][1]cuda:0", arg104_1: "bf16[1024, 1024][1024, 1]cuda:0", arg105_1: "bf16[1024][1]cuda:0", arg106_1: "bf16[1024, 1024][1024, 1]cuda:0", arg107_1: "bf16[1024][1]cuda:0", arg108_1: "bf16[1024, 1024][1024, 1]cuda:0", arg109_1: "bf16[1024][1]cuda:0", arg110_1: "bf16[1024][1]cuda:0", arg111_1: "bf16[1024][1]cuda:0", arg112_1: "bf16[4096, 1024][1024, 1]cuda:0", arg113_1: "bf16[4096][1]cuda:0", arg114_1: "bf16[1024, 4096][4096, 1]cuda:0", arg115_1: "bf16[1024][1]cuda:0", arg116_1: "bf16[1024][1]cuda:0", arg117_1: "bf16[1024][1]cuda:0", arg118_1: "bf16[1024, 1024][1024, 1]cuda:0", arg119_1: "bf16[1024][1]cuda:0", arg120_1: "bf16[1024, 1024][1024, 1]cuda:0", arg121_1: "bf16[1024][1]cuda:0", arg122_1: "bf16[1024, 1024][1024, 1]cuda:0", arg123_1: "bf16[1024][1]cuda:0", arg124_1: "bf16[1024, 1024][1024, 1]cuda:0", arg125_1: "bf16[1024][1]cuda:0", arg126_1: "bf16[1024][1]cuda:0", arg127_1: "bf16[1024][1]cuda:0", arg128_1: "bf16[4096, 1024][1024, 1]cuda:0", arg129_1: "bf16[4096][1]cuda:0", arg130_1: "bf16[1024, 4096][4096, 1]cuda:0", arg131_1: "bf16[1024][1]cuda:0", arg132_1: "bf16[1024][1]cuda:0", arg133_1: "bf16[1024][1]cuda:0", arg134_1: "bf16[1024, 1024][1024, 1]cuda:0", arg135_1: "bf16[1024][1]cuda:0", arg136_1: "bf16[1024, 1024][1024, 1]cuda:0", arg137_1: "bf16[1024][1]cuda:0", arg138_1: "bf16[1024, 1024][1024, 1]cuda:0", arg139_1: "bf16[1024][1]cuda:0", arg140_1: "bf16[1024, 1024][1024, 1]cuda:0", arg141_1: "bf16[1024][1]cuda:0", arg142_1: "bf16[1024][1]cuda:0", arg143_1: "bf16[1024][1]cuda:0", arg144_1: "bf16[4096, 1024][1024, 1]cuda:0", arg145_1: "bf16[4096][1]cuda:0", arg146_1: "bf16[1024, 4096][4096, 1]cuda:0", arg147_1: "bf16[1024][1]cuda:0", arg148_1: "bf16[1024][1]cuda:0", arg149_1: "bf16[1024][1]cuda:0", arg150_1: "bf16[1024, 1024][1024, 1]cuda:0", arg151_1: "bf16[1024][1]cuda:0", arg152_1: "bf16[1024, 1024][1024, 1]cuda:0", arg153_1: "bf16[1024][1]cuda:0", arg154_1: "bf16[1024, 1024][1024, 1]cuda:0", arg155_1: "bf16[1024][1]cuda:0", arg156_1: "bf16[1024, 1024][1024, 1]cuda:0", arg157_1: "bf16[1024][1]cuda:0", arg158_1: "bf16[1024][1]cuda:0", arg159_1: "bf16[1024][1]cuda:0", arg160_1: "bf16[4096, 1024][1024, 1]cuda:0", arg161_1: "bf16[4096][1]cuda:0", arg162_1: "bf16[1024, 4096][4096, 1]cuda:0", arg163_1: "bf16[1024][1]cuda:0", arg164_1: "bf16[1024][1]cuda:0", arg165_1: "bf16[1024][1]cuda:0", arg166_1: "bf16[1024, 1024][1024, 1]cuda:0", arg167_1: "bf16[1024][1]cuda:0", arg168_1: "bf16[1024, 1024][1024, 1]cuda:0", arg169_1: "bf16[1024][1]cuda:0", arg170_1: "bf16[1024, 1024][1024, 1]cuda:0", arg171_1: "bf16[1024][1]cuda:0", arg172_1: "bf16[1024, 1024][1024, 1]cuda:0", arg173_1: "bf16[1024][1]cuda:0", arg174_1: "bf16[1024][1]cuda:0", arg175_1: "bf16[1024][1]cuda:0", arg176_1: "bf16[4096, 1024][1024, 1]cuda:0", arg177_1: "bf16[4096][1]cuda:0", arg178_1: "bf16[1024, 4096][4096, 1]cuda:0", arg179_1: "bf16[1024][1]cuda:0", arg180_1: "bf16[1024][1]cuda:0", arg181_1: "bf16[1024][1]cuda:0", arg182_1: "bf16[1024, 1024][1024, 1]cuda:0", arg183_1: "bf16[1024][1]cuda:0", arg184_1: "bf16[1024, 1024][1024, 1]cuda:0", arg185_1: "bf16[1024][1]cuda:0", arg186_1: "bf16[1024, 1024][1024, 1]cuda:0", arg187_1: "bf16[1024][1]cuda:0", arg188_1: "bf16[1024, 1024][1024, 1]cuda:0", arg189_1: "bf16[1024][1]cuda:0", arg190_1: "bf16[1024][1]cuda:0", arg191_1: "bf16[1024][1]cuda:0", arg192_1: "bf16[4096, 1024][1024, 1]cuda:0", arg193_1: "bf16[4096][1]cuda:0", arg194_1: "bf16[1024, 4096][4096, 1]cuda:0", arg195_1: "bf16[1024][1]cuda:0", arg196_1: "bf16[1024][1]cuda:0", arg197_1: "bf16[1024][1]cuda:0", arg198_1: "bf16[1026, 1024][1024, 1]cuda:0", arg199_1: "bf16[1024][1]cuda:0", arg200_1: "bf16[1024][1]cuda:0", arg201_1: "bf16[1024, 1024][1024, 1]cuda:0", arg202_1: "bf16[1024][1]cuda:0", arg203_1: "bf16[1024, 1024][1024, 1]cuda:0", arg204_1: "bf16[1024][1]cuda:0", arg205_1: "bf16[1024, 1024][1024, 1]cuda:0", arg206_1: "bf16[1024][1]cuda:0", arg207_1: "bf16[1024, 1024][1024, 1]cuda:0", arg208_1: "bf16[1024][1]cuda:0", arg209_1: "bf16[1024][1]cuda:0", arg210_1: "bf16[1024][1]cuda:0", arg211_1: "bf16[1024, 1024][1024, 1]cuda:0", arg212_1: "bf16[1024][1]cuda:0", arg213_1: "bf16[1024, 1024][1024, 1]cuda:0", arg214_1: "bf16[1024][1]cuda:0", arg215_1: "bf16[1024, 1024][1024, 1]cuda:0", arg216_1: "bf16[1024][1]cuda:0", arg217_1: "bf16[1024, 1024][1024, 1]cuda:0", arg218_1: "bf16[1024][1]cuda:0", arg219_1: "bf16[1024][1]cuda:0", arg220_1: "bf16[1024][1]cuda:0", arg221_1: "bf16[4096, 1024][1024, 1]cuda:0", arg222_1: "bf16[4096][1]cuda:0", arg223_1: "bf16[1024, 4096][4096, 1]cuda:0", arg224_1: "bf16[1024][1]cuda:0", arg225_1: "bf16[1024][1]cuda:0", arg226_1: "bf16[1024][1]cuda:0", arg227_1: "bf16[1024, 1024][1024, 1]cuda:0", arg228_1: "bf16[1024][1]cuda:0", arg229_1: "bf16[1024, 1024][1024, 1]cuda:0", arg230_1: "bf16[1024][1]cuda:0", arg231_1: "bf16[1024, 1024][1024, 1]cuda:0", arg232_1: "bf16[1024][1]cuda:0", arg233_1: "bf16[1024, 1024][1024, 1]cuda:0", arg234_1: "bf16[1024][1]cuda:0", arg235_1: "bf16[1024][1]cuda:0", arg236_1: "bf16[1024][1]cuda:0", arg237_1: "bf16[1024, 1024][1024, 1]cuda:0", arg238_1: "bf16[1024][1]cuda:0", arg239_1: "bf16[1024, 1024][1024, 1]cuda:0", arg240_1: "bf16[1024][1]cuda:0", arg241_1: "bf16[1024, 1024][1024, 1]cuda:0", arg242_1: "bf16[1024][1]cuda:0", arg243_1: "bf16[1024, 1024][1024, 1]cuda:0", arg244_1: "bf16[1024][1]cuda:0", arg245_1: "bf16[1024][1]cuda:0", arg246_1: "bf16[1024][1]cuda:0", arg247_1: "bf16[4096, 1024][1024, 1]cuda:0", arg248_1: "bf16[4096][1]cuda:0", arg249_1: "bf16[1024, 4096][4096, 1]cuda:0", arg250_1: "bf16[1024][1]cuda:0", arg251_1: "bf16[1024][1]cuda:0", arg252_1: "bf16[1024][1]cuda:0", arg253_1: "bf16[1024, 1024][1024, 1]cuda:0", arg254_1: "bf16[1024][1]cuda:0", arg255_1: "bf16[1024, 1024][1024, 1]cuda:0", arg256_1: "bf16[1024][1]cuda:0", arg257_1: "bf16[1024, 1024][1024, 1]cuda:0", arg258_1: "bf16[1024][1]cuda:0", arg259_1: "bf16[1024, 1024][1024, 1]cuda:0", arg260_1: "bf16[1024][1]cuda:0", arg261_1: "bf16[1024][1]cuda:0", arg262_1: "bf16[1024][1]cuda:0", arg263_1: "bf16[1024, 1024][1024, 1]cuda:0", arg264_1: "bf16[1024][1]cuda:0", arg265_1: "bf16[1024, 1024][1024, 1]cuda:0", arg266_1: "bf16[1024][1]cuda:0", arg267_1: "bf16[1024, 1024][1024, 1]cuda:0", arg268_1: "bf16[1024][1]cuda:0", arg269_1: "bf16[1024, 1024][1024, 1]cuda:0", arg270_1: "bf16[1024][1]cuda:0", arg271_1: "bf16[1024][1]cuda:0", arg272_1: "bf16[1024][1]cuda:0", arg273_1: "bf16[4096, 1024][1024, 1]cuda:0", arg274_1: "bf16[4096][1]cuda:0", arg275_1: "bf16[1024, 4096][4096, 1]cuda:0", arg276_1: "bf16[1024][1]cuda:0", arg277_1: "bf16[1024][1]cuda:0", arg278_1: "bf16[1024][1]cuda:0", arg279_1: "bf16[1024, 1024][1024, 1]cuda:0", arg280_1: "bf16[1024][1]cuda:0", arg281_1: "bf16[1024, 1024][1024, 1]cuda:0", arg282_1: "bf16[1024][1]cuda:0", arg283_1: "bf16[1024, 1024][1024, 1]cuda:0", arg284_1: "bf16[1024][1]cuda:0", arg285_1: "bf16[1024, 1024][1024, 1]cuda:0", arg286_1: "bf16[1024][1]cuda:0", arg287_1: "bf16[1024][1]cuda:0", arg288_1: "bf16[1024][1]cuda:0", arg289_1: "bf16[1024, 1024][1024, 1]cuda:0", arg290_1: "bf16[1024][1]cuda:0", arg291_1: "bf16[1024, 1024][1024, 1]cuda:0", arg292_1: "bf16[1024][1]cuda:0", arg293_1: "bf16[1024, 1024][1024, 1]cuda:0", arg294_1: "bf16[1024][1]cuda:0", arg295_1: "bf16[1024, 1024][1024, 1]cuda:0", arg296_1: "bf16[1024][1]cuda:0", arg297_1: "bf16[1024][1]cuda:0", arg298_1: "bf16[1024][1]cuda:0", arg299_1: "bf16[4096, 1024][1024, 1]cuda:0", arg300_1: "bf16[4096][1]cuda:0", arg301_1: "bf16[1024, 4096][4096, 1]cuda:0", arg302_1: "bf16[1024][1]cuda:0", arg303_1: "bf16[1024][1]cuda:0", arg304_1: "bf16[1024][1]cuda:0", arg305_1: "bf16[1024, 1024][1024, 1]cuda:0", arg306_1: "bf16[1024][1]cuda:0", arg307_1: "bf16[1024, 1024][1024, 1]cuda:0", arg308_1: "bf16[1024][1]cuda:0", arg309_1: "bf16[1024, 1024][1024, 1]cuda:0", arg310_1: "bf16[1024][1]cuda:0", arg311_1: "bf16[1024, 1024][1024, 1]cuda:0", arg312_1: "bf16[1024][1]cuda:0", arg313_1: "bf16[1024][1]cuda:0", arg314_1: "bf16[1024][1]cuda:0", arg315_1: "bf16[1024, 1024][1024, 1]cuda:0", arg316_1: "bf16[1024][1]cuda:0", arg317_1: "bf16[1024, 1024][1024, 1]cuda:0", arg318_1: "bf16[1024][1]cuda:0", arg319_1: "bf16[1024, 1024][1024, 1]cuda:0", arg320_1: "bf16[1024][1]cuda:0", arg321_1: "bf16[1024, 1024][1024, 1]cuda:0", arg322_1: "bf16[1024][1]cuda:0", arg323_1: "bf16[1024][1]cuda:0", arg324_1: "bf16[1024][1]cuda:0", arg325_1: "bf16[4096, 1024][1024, 1]cuda:0", arg326_1: "bf16[4096][1]cuda:0", arg327_1: "bf16[1024, 4096][4096, 1]cuda:0", arg328_1: "bf16[1024][1]cuda:0", arg329_1: "bf16[1024][1]cuda:0", arg330_1: "bf16[1024][1]cuda:0", arg331_1: "bf16[1024, 1024][1024, 1]cuda:0", arg332_1: "bf16[1024][1]cuda:0", arg333_1: "bf16[1024, 1024][1024, 1]cuda:0", arg334_1: "bf16[1024][1]cuda:0", arg335_1: "bf16[1024, 1024][1024, 1]cuda:0", arg336_1: "bf16[1024][1]cuda:0", arg337_1: "bf16[1024, 1024][1024, 1]cuda:0", arg338_1: "bf16[1024][1]cuda:0", arg339_1: "bf16[1024][1]cuda:0", arg340_1: "bf16[1024][1]cuda:0", arg341_1: "bf16[1024, 1024][1024, 1]cuda:0", arg342_1: "bf16[1024][1]cuda:0", arg343_1: "bf16[1024, 1024][1024, 1]cuda:0", arg344_1: "bf16[1024][1]cuda:0", arg345_1: "bf16[1024, 1024][1024, 1]cuda:0", arg346_1: "bf16[1024][1]cuda:0", arg347_1: "bf16[1024, 1024][1024, 1]cuda:0", arg348_1: "bf16[1024][1]cuda:0", arg349_1: "bf16[1024][1]cuda:0", arg350_1: "bf16[1024][1]cuda:0", arg351_1: "bf16[4096, 1024][1024, 1]cuda:0", arg352_1: "bf16[4096][1]cuda:0", arg353_1: "bf16[1024, 4096][4096, 1]cuda:0", arg354_1: "bf16[1024][1]cuda:0", arg355_1: "bf16[1024][1]cuda:0", arg356_1: "bf16[1024][1]cuda:0", arg357_1: "bf16[1024, 1024][1024, 1]cuda:0", arg358_1: "bf16[1024][1]cuda:0", arg359_1: "bf16[1024, 1024][1024, 1]cuda:0", arg360_1: "bf16[1024][1]cuda:0", arg361_1: "bf16[1024, 1024][1024, 1]cuda:0", arg362_1: "bf16[1024][1]cuda:0", arg363_1: "bf16[1024, 1024][1024, 1]cuda:0", arg364_1: "bf16[1024][1]cuda:0", arg365_1: "bf16[1024][1]cuda:0", arg366_1: "bf16[1024][1]cuda:0", arg367_1: "bf16[1024, 1024][1024, 1]cuda:0", arg368_1: "bf16[1024][1]cuda:0", arg369_1: "bf16[1024, 1024][1024, 1]cuda:0", arg370_1: "bf16[1024][1]cuda:0", arg371_1: "bf16[1024, 1024][1024, 1]cuda:0", arg372_1: "bf16[1024][1]cuda:0", arg373_1: "bf16[1024, 1024][1024, 1]cuda:0", arg374_1: "bf16[1024][1]cuda:0", arg375_1: "bf16[1024][1]cuda:0", arg376_1: "bf16[1024][1]cuda:0", arg377_1: "bf16[4096, 1024][1024, 1]cuda:0", arg378_1: "bf16[4096][1]cuda:0", arg379_1: "bf16[1024, 4096][4096, 1]cuda:0", arg380_1: "bf16[1024][1]cuda:0", arg381_1: "bf16[1024][1]cuda:0", arg382_1: "bf16[1024][1]cuda:0", arg383_1: "bf16[1024, 1024][1024, 1]cuda:0", arg384_1: "bf16[1024][1]cuda:0", arg385_1: "bf16[1024, 1024][1024, 1]cuda:0", arg386_1: "bf16[1024][1]cuda:0", arg387_1: "bf16[1024, 1024][1024, 1]cuda:0", arg388_1: "bf16[1024][1]cuda:0", arg389_1: "bf16[1024, 1024][1024, 1]cuda:0", arg390_1: "bf16[1024][1]cuda:0", arg391_1: "bf16[1024][1]cuda:0", arg392_1: "bf16[1024][1]cuda:0", arg393_1: "bf16[1024, 1024][1024, 1]cuda:0", arg394_1: "bf16[1024][1]cuda:0", arg395_1: "bf16[1024, 1024][1024, 1]cuda:0", arg396_1: "bf16[1024][1]cuda:0", arg397_1: "bf16[1024, 1024][1024, 1]cuda:0", arg398_1: "bf16[1024][1]cuda:0", arg399_1: "bf16[1024, 1024][1024, 1]cuda:0", arg400_1: "bf16[1024][1]cuda:0", arg401_1: "bf16[1024][1]cuda:0", arg402_1: "bf16[1024][1]cuda:0", arg403_1: "bf16[4096, 1024][1024, 1]cuda:0", arg404_1: "bf16[4096][1]cuda:0", arg405_1: "bf16[1024, 4096][4096, 1]cuda:0", arg406_1: "bf16[1024][1]cuda:0", arg407_1: "bf16[1024][1]cuda:0", arg408_1: "bf16[1024][1]cuda:0", arg409_1: "bf16[1024, 1024][1024, 1]cuda:0", arg410_1: "bf16[1024][1]cuda:0", arg411_1: "bf16[1024, 1024][1024, 1]cuda:0", arg412_1: "bf16[1024][1]cuda:0", arg413_1: "bf16[1024, 1024][1024, 1]cuda:0", arg414_1: "bf16[1024][1]cuda:0", arg415_1: "bf16[1024, 1024][1024, 1]cuda:0", arg416_1: "bf16[1024][1]cuda:0", arg417_1: "bf16[1024][1]cuda:0", arg418_1: "bf16[1024][1]cuda:0", arg419_1: "bf16[1024, 1024][1024, 1]cuda:0", arg420_1: "bf16[1024][1]cuda:0", arg421_1: "bf16[1024, 1024][1024, 1]cuda:0", arg422_1: "bf16[1024][1]cuda:0", arg423_1: "bf16[1024, 1024][1024, 1]cuda:0", arg424_1: "bf16[1024][1]cuda:0", arg425_1: "bf16[1024, 1024][1024, 1]cuda:0", arg426_1: "bf16[1024][1]cuda:0", arg427_1: "bf16[1024][1]cuda:0", arg428_1: "bf16[1024][1]cuda:0", arg429_1: "bf16[4096, 1024][1024, 1]cuda:0", arg430_1: "bf16[4096][1]cuda:0", arg431_1: "bf16[1024, 4096][4096, 1]cuda:0", arg432_1: "bf16[1024][1]cuda:0", arg433_1: "bf16[1024][1]cuda:0", arg434_1: "bf16[1024][1]cuda:0", arg435_1: "bf16[1024, 1024][1024, 1]cuda:0", arg436_1: "bf16[1024][1]cuda:0", arg437_1: "bf16[1024, 1024][1024, 1]cuda:0", arg438_1: "bf16[1024][1]cuda:0", arg439_1: "bf16[1024, 1024][1024, 1]cuda:0", arg440_1: "bf16[1024][1]cuda:0", arg441_1: "bf16[1024, 1024][1024, 1]cuda:0", arg442_1: "bf16[1024][1]cuda:0", arg443_1: "bf16[1024][1]cuda:0", arg444_1: "bf16[1024][1]cuda:0", arg445_1: "bf16[1024, 1024][1024, 1]cuda:0", arg446_1: "bf16[1024][1]cuda:0", arg447_1: "bf16[1024, 1024][1024, 1]cuda:0", arg448_1: "bf16[1024][1]cuda:0", arg449_1: "bf16[1024, 1024][1024, 1]cuda:0", arg450_1: "bf16[1024][1]cuda:0", arg451_1: "bf16[1024, 1024][1024, 1]cuda:0", arg452_1: "bf16[1024][1]cuda:0", arg453_1: "bf16[1024][1]cuda:0", arg454_1: "bf16[1024][1]cuda:0", arg455_1: "bf16[4096, 1024][1024, 1]cuda:0", arg456_1: "bf16[4096][1]cuda:0", arg457_1: "bf16[1024, 4096][4096, 1]cuda:0", arg458_1: "bf16[1024][1]cuda:0", arg459_1: "bf16[1024][1]cuda:0", arg460_1: "bf16[1024][1]cuda:0", arg461_1: "bf16[1024, 1024][1024, 1]cuda:0", arg462_1: "bf16[1024][1]cuda:0", arg463_1: "bf16[1024, 1024][1024, 1]cuda:0", arg464_1: "bf16[1024][1]cuda:0", arg465_1: "bf16[1024, 1024][1024, 1]cuda:0", arg466_1: "bf16[1024][1]cuda:0", arg467_1: "bf16[1024, 1024][1024, 1]cuda:0", arg468_1: "bf16[1024][1]cuda:0", arg469_1: "bf16[1024][1]cuda:0", arg470_1: "bf16[1024][1]cuda:0", arg471_1: "bf16[1024, 1024][1024, 1]cuda:0", arg472_1: "bf16[1024][1]cuda:0", arg473_1: "bf16[1024, 1024][1024, 1]cuda:0", arg474_1: "bf16[1024][1]cuda:0", arg475_1: "bf16[1024, 1024][1024, 1]cuda:0", arg476_1: "bf16[1024][1]cuda:0", arg477_1: "bf16[1024, 1024][1024, 1]cuda:0", arg478_1: "bf16[1024][1]cuda:0", arg479_1: "bf16[1024][1]cuda:0", arg480_1: "bf16[1024][1]cuda:0", arg481_1: "bf16[4096, 1024][1024, 1]cuda:0", arg482_1: "bf16[4096][1]cuda:0", arg483_1: "bf16[1024, 4096][4096, 1]cuda:0", arg484_1: "bf16[1024][1]cuda:0", arg485_1: "bf16[1024][1]cuda:0", arg486_1: "bf16[1024][1]cuda:0", arg487_1: "bf16[1024, 1024][1024, 1]cuda:0", arg488_1: "bf16[1024][1]cuda:0", arg489_1: "bf16[1024, 1024][1024, 1]cuda:0", arg490_1: "bf16[1024][1]cuda:0", arg491_1: "bf16[1024, 1024][1024, 1]cuda:0", arg492_1: "bf16[1024][1]cuda:0", arg493_1: "bf16[1024, 1024][1024, 1]cuda:0", arg494_1: "bf16[1024][1]cuda:0", arg495_1: "bf16[1024][1]cuda:0", arg496_1: "bf16[1024][1]cuda:0", arg497_1: "bf16[1024, 1024][1024, 1]cuda:0", arg498_1: "bf16[1024][1]cuda:0", arg499_1: "bf16[1024, 1024][1024, 1]cuda:0", arg500_1: "bf16[1024][1]cuda:0", arg501_1: "bf16[1024, 1024][1024, 1]cuda:0", arg502_1: "bf16[1024][1]cuda:0", arg503_1: "bf16[1024, 1024][1024, 1]cuda:0", arg504_1: "bf16[1024][1]cuda:0", arg505_1: "bf16[1024][1]cuda:0", arg506_1: "bf16[1024][1]cuda:0", arg507_1: "bf16[4096, 1024][1024, 1]cuda:0", arg508_1: "bf16[4096][1]cuda:0", arg509_1: "bf16[1024, 4096][4096, 1]cuda:0", arg510_1: "bf16[1024][1]cuda:0", arg511_1: "bf16[1024][1]cuda:0", arg512_1: "bf16[1024][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_1: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[64, 1, 128, 128][0, 128, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge, [64, -1, 128, 128]);  ge = expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:77 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 1)
        mul: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:177 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne: "b8[64, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg1_1, 1)
        convert_element_type: "i32[64, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:178 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        cumsum: "i64[64, 128][128, 1]cuda:0" = torch.ops.aten.cumsum.default(convert_element_type, 1)
        convert_element_type_1: "i32[64, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add: "i32[64, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
        mul_1: "i32[64, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(add, convert_element_type);  add = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:179 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_2: "i64[64, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        add_1: "i64[64, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:144 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        view: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(add_1, [-1]);  add_1 = None
        index: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.index.Tensor(arg3_1, [view]);  arg3_1 = view = None
        view_1: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(index, [64, 128, 1024]);  index = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:562 in forward, code: hidden_states = inputs_embeds + embed_pos
        add_2: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, view_1);  mul = view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_3: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_3, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor = None
        scalar_tensor_1: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_1 = None
        full_default: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_3, getitem_1);  convert_element_type_3 = getitem_1 = None
        add_5: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_2: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_3: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        add_6: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
        convert_element_type_4: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_4, [8192, 1024])
        permute: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg7_1, view_2, permute);  arg7_1 = view_2 = permute = None
        view_3: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [64, 128, 1024]);  addmm = None
        view_4: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [64, 128, -1, 64]);  view_3 = None
        permute_1: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_4: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        expand_1: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_4, [64, 16, 128, 64]);  mul_4 = None
        clone_1: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_11: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [1024, 128, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_5: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_4, [8192, 1024])
        permute_2: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg9_1, view_5, permute_2);  arg9_1 = view_5 = permute_2 = None
        view_6: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [64, 128, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_9: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [64, 128, -1, 64]);  view_6 = None
        permute_4: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_6: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        mul_5: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_2: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_5, [64, 16, 64, 128]);  mul_5 = None
        clone_2: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_12: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1024, 64, 128]);  clone_2 = None
        bmm: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_11, view_12);  view_11 = view_12 = None
        view_13: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [64, 16, 128, 128]);  bmm = None
        eq: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_13, -inf)
        logical_not: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_1: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_16: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        amax: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_16, [-1], True)
        sub_1: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_16, amax);  convert_element_type_16 = amax = None
        exp: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_17: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where_1: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_1, convert_element_type_17);  logical_not_1 = full_default_1 = convert_element_type_17 = None
        expand_3: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_1, [64, 16, 128, 128]);  where_1 = None
        view_14: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_3, [1024, 128, 128]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_7: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_4, [8192, 1024]);  convert_element_type_4 = None
        permute_3: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg11_1, view_7, permute_3);  arg11_1 = view_7 = permute_3 = None
        view_8: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [64, 128, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_10: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_8, [64, 128, -1, 64]);  view_8 = None
        permute_5: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [64, 16, 128, 64]);  permute_5 = None
        clone_3: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_15: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [1024, 128, 64]);  clone_3 = None
        bmm_1: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = view_15 = None
        view_16: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [64, 16, 128, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_4: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_17: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [64, 128, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_17, [8192, 1024]);  view_17 = None
        permute_8: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg13_1, view_18, permute_8);  arg13_1 = view_18 = permute_8 = None
        view_19: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [64, 128, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_8: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_2, view_19);  add_2 = view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_23: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_23, [2], correction = 0, keepdim = True)
        getitem_2: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, getitem_3);  convert_element_type_23 = getitem_3 = None
        add_9: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_6: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_7: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, arg14_1);  mul_6 = arg14_1 = None
        add_10: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, arg15_1);  mul_7 = arg15_1 = None
        convert_element_type_24: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_20: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_24, [8192, 1024]);  convert_element_type_24 = None
        permute_9: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg17_1, view_20, permute_9);  arg17_1 = view_20 = permute_9 = None
        view_21: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [64, 128, 4096]);  addmm_4 = None
        relu: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_21);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_22: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu, [8192, 4096]);  relu = None
        permute_10: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg19_1, view_22, permute_10);  arg19_1 = view_22 = permute_10 = None
        view_23: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [64, 128, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_11: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_8, view_23);  add_8 = view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_31: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_31, [2], correction = 0, keepdim = True)
        getitem_4: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_2: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_2 = None
        scalar_tensor_3: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_3 = None
        full_default_2: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_3: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_31, getitem_5);  convert_element_type_31 = getitem_5 = None
        add_12: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_8: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_9: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, arg20_1);  mul_8 = arg20_1 = None
        add_13: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, arg21_1);  mul_9 = arg21_1 = None
        convert_element_type_32: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_24: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_32, [8192, 1024])
        permute_11: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg23_1, view_24, permute_11);  arg23_1 = view_24 = permute_11 = None
        view_25: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [64, 128, 1024]);  addmm_6 = None
        view_26: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [64, 128, -1, 64]);  view_25 = None
        permute_12: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_10: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        expand_5: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_10, [64, 16, 128, 64]);  mul_10 = None
        clone_8: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_33: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1024, 128, 64]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_27: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_32, [8192, 1024])
        permute_13: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg25_1, view_27, permute_13);  arg25_1 = view_27 = permute_13 = None
        view_28: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [64, 128, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_31: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_28, [64, 128, -1, 64]);  view_28 = None
        permute_15: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_17: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        mul_11: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        expand_6: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_11, [64, 16, 64, 128]);  mul_11 = None
        clone_9: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_34: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1024, 64, 128]);  clone_9 = None
        bmm_2: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_33, view_34);  view_33 = view_34 = None
        view_35: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [64, 16, 128, 128]);  bmm_2 = None
        eq_1: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_35, -inf)
        logical_not_2: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_3: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_44: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        amax_1: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_44, [-1], True)
        sub_4: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_44, amax_1);  convert_element_type_44 = amax_1 = None
        exp_1: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_45: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        where_3: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_3, full_default_3, convert_element_type_45);  logical_not_3 = full_default_3 = convert_element_type_45 = None
        expand_7: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_3, [64, 16, 128, 128]);  where_3 = None
        view_36: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_7, [1024, 128, 128]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_29: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_32, [8192, 1024]);  convert_element_type_32 = None
        permute_14: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg27_1, view_29, permute_14);  arg27_1 = view_29 = permute_14 = None
        view_30: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [64, 128, 1024]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_32: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [64, 128, -1, 64]);  view_30 = None
        permute_16: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_16, [64, 16, 128, 64]);  permute_16 = None
        clone_10: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_37: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1024, 128, 64]);  clone_10 = None
        bmm_3: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
        view_38: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [64, 16, 128, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_11: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_39: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [64, 128, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_40: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_39, [8192, 1024]);  view_39 = None
        permute_19: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg29_1, view_40, permute_19);  arg29_1 = view_40 = permute_19 = None
        view_41: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [64, 128, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_15: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, view_41);  add_11 = view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_51: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_51, [2], correction = 0, keepdim = True)
        getitem_6: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_51, getitem_7);  convert_element_type_51 = getitem_7 = None
        add_16: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_12: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, arg30_1);  mul_12 = arg30_1 = None
        add_17: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, arg31_1);  mul_13 = arg31_1 = None
        convert_element_type_52: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_42: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_52, [8192, 1024]);  convert_element_type_52 = None
        permute_20: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg33_1, view_42, permute_20);  arg33_1 = view_42 = permute_20 = None
        view_43: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [64, 128, 4096]);  addmm_10 = None
        relu_1: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_43);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_44: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_1, [8192, 4096]);  relu_1 = None
        permute_21: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg35_1, view_44, permute_21);  arg35_1 = view_44 = permute_21 = None
        view_45: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [64, 128, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_18: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_15, view_45);  add_15 = view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_59: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_59, [2], correction = 0, keepdim = True)
        getitem_8: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_4: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_4 = None
        scalar_tensor_5: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_5 = None
        full_default_4: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_6: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_59, getitem_9);  convert_element_type_59 = getitem_9 = None
        add_19: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_14: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_15: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, arg36_1);  mul_14 = arg36_1 = None
        add_20: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, arg37_1);  mul_15 = arg37_1 = None
        convert_element_type_60: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_46: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_60, [8192, 1024])
        permute_22: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg39_1, view_46, permute_22);  arg39_1 = view_46 = permute_22 = None
        view_47: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [64, 128, 1024]);  addmm_12 = None
        view_48: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_47, [64, 128, -1, 64]);  view_47 = None
        permute_23: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_16: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_23, 0.3535533905932738);  permute_23 = None
        expand_9: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_16, [64, 16, 128, 64]);  mul_16 = None
        clone_15: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_55: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [1024, 128, 64]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_49: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_60, [8192, 1024])
        permute_24: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg41_1, view_49, permute_24);  arg41_1 = view_49 = permute_24 = None
        view_50: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [64, 128, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_53: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [64, 128, -1, 64]);  view_50 = None
        permute_26: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_28: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        mul_17: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
        expand_10: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_17, [64, 16, 64, 128]);  mul_17 = None
        clone_16: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_56: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [1024, 64, 128]);  clone_16 = None
        bmm_4: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
        view_57: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [64, 16, 128, 128]);  bmm_4 = None
        eq_2: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_57, -inf)
        logical_not_4: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_5: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_72: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        amax_2: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_72, [-1], True)
        sub_7: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, amax_2);  convert_element_type_72 = amax_2 = None
        exp_2: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_73: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        where_5: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_5, full_default_5, convert_element_type_73);  logical_not_5 = full_default_5 = convert_element_type_73 = None
        expand_11: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_5, [64, 16, 128, 128]);  where_5 = None
        view_58: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_11, [1024, 128, 128]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_51: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_60, [8192, 1024]);  convert_element_type_60 = None
        permute_25: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg43_1, view_51, permute_25);  arg43_1 = view_51 = permute_25 = None
        view_52: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [64, 128, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_54: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [64, 128, -1, 64]);  view_52 = None
        permute_27: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_12: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_27, [64, 16, 128, 64]);  permute_27 = None
        clone_17: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_59: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1024, 128, 64]);  clone_17 = None
        bmm_5: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [64, 16, 128, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_18: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [64, 128, -1]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_62: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_61, [8192, 1024]);  view_61 = None
        permute_30: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg45_1, view_62, permute_30);  arg45_1 = view_62 = permute_30 = None
        view_63: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [64, 128, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_22: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_18, view_63);  add_18 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_79: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_79, [2], correction = 0, keepdim = True)
        getitem_10: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_79, getitem_11);  convert_element_type_79 = getitem_11 = None
        add_23: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_18: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_19: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, arg46_1);  mul_18 = arg46_1 = None
        add_24: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, arg47_1);  mul_19 = arg47_1 = None
        convert_element_type_80: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_64: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_80, [8192, 1024]);  convert_element_type_80 = None
        permute_31: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg49_1, view_64, permute_31);  arg49_1 = view_64 = permute_31 = None
        view_65: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [64, 128, 4096]);  addmm_16 = None
        relu_2: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_65);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_66: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_2, [8192, 4096]);  relu_2 = None
        permute_32: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg51_1, view_66, permute_32);  arg51_1 = view_66 = permute_32 = None
        view_67: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [64, 128, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_25: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_22, view_67);  add_22 = view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_87: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_87, [2], correction = 0, keepdim = True)
        getitem_12: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_6: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_6 = None
        scalar_tensor_7: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_7 = None
        full_default_6: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_9: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_87, getitem_13);  convert_element_type_87 = getitem_13 = None
        add_26: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_20: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_21: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, arg52_1);  mul_20 = arg52_1 = None
        add_27: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, arg53_1);  mul_21 = arg53_1 = None
        convert_element_type_88: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_68: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_88, [8192, 1024])
        permute_33: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg55_1, view_68, permute_33);  arg55_1 = view_68 = permute_33 = None
        view_69: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [64, 128, 1024]);  addmm_18 = None
        view_70: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_69, [64, 128, -1, 64]);  view_69 = None
        permute_34: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_22: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_34, 0.3535533905932738);  permute_34 = None
        expand_13: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_22, [64, 16, 128, 64]);  mul_22 = None
        clone_22: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_77: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [1024, 128, 64]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_71: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_88, [8192, 1024])
        permute_35: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg57_1, view_71, permute_35);  arg57_1 = view_71 = permute_35 = None
        view_72: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [64, 128, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_75: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_72, [64, 128, -1, 64]);  view_72 = None
        permute_37: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_39: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        mul_23: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_39, 0.3535533905932738);  permute_39 = None
        expand_14: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_23, [64, 16, 64, 128]);  mul_23 = None
        clone_23: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_78: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [1024, 64, 128]);  clone_23 = None
        bmm_6: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_77, view_78);  view_77 = view_78 = None
        view_79: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [64, 16, 128, 128]);  bmm_6 = None
        eq_3: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_79, -inf)
        logical_not_6: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_7: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_100: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        amax_3: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_100, [-1], True)
        sub_10: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_100, amax_3);  convert_element_type_100 = amax_3 = None
        exp_3: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_101: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        where_7: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_7, full_default_7, convert_element_type_101);  logical_not_7 = full_default_7 = convert_element_type_101 = None
        expand_15: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_7, [64, 16, 128, 128]);  where_7 = None
        view_80: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_15, [1024, 128, 128]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_73: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_88, [8192, 1024]);  convert_element_type_88 = None
        permute_36: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg59_1, view_73, permute_36);  arg59_1 = view_73 = permute_36 = None
        view_74: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [64, 128, 1024]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_76: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [64, 128, -1, 64]);  view_74 = None
        permute_38: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_16: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_38, [64, 16, 128, 64]);  permute_38 = None
        clone_24: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_81: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [1024, 128, 64]);  clone_24 = None
        bmm_7: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
        view_82: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [64, 16, 128, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_25: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_83: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [64, 128, -1]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_84: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_83, [8192, 1024]);  view_83 = None
        permute_41: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg61_1, view_84, permute_41);  arg61_1 = view_84 = permute_41 = None
        view_85: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [64, 128, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_29: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_25, view_85);  add_25 = view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_107: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_107, [2], correction = 0, keepdim = True)
        getitem_14: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_107, getitem_15);  convert_element_type_107 = getitem_15 = None
        add_30: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_24: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_25: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, arg62_1);  mul_24 = arg62_1 = None
        add_31: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, arg63_1);  mul_25 = arg63_1 = None
        convert_element_type_108: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_86: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_108, [8192, 1024]);  convert_element_type_108 = None
        permute_42: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg65_1, view_86, permute_42);  arg65_1 = view_86 = permute_42 = None
        view_87: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [64, 128, 4096]);  addmm_22 = None
        relu_3: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_87);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_88: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_3, [8192, 4096]);  relu_3 = None
        permute_43: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg67_1, view_88, permute_43);  arg67_1 = view_88 = permute_43 = None
        view_89: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [64, 128, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_32: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_29, view_89);  add_29 = view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_115: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_115, [2], correction = 0, keepdim = True)
        getitem_16: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_8: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_8 = None
        scalar_tensor_9: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_9 = None
        full_default_8: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_12: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_115, getitem_17);  convert_element_type_115 = getitem_17 = None
        add_33: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_26: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_27: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, arg68_1);  mul_26 = arg68_1 = None
        add_34: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, arg69_1);  mul_27 = arg69_1 = None
        convert_element_type_116: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_90: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [8192, 1024])
        permute_44: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg71_1, view_90, permute_44);  arg71_1 = view_90 = permute_44 = None
        view_91: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [64, 128, 1024]);  addmm_24 = None
        view_92: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_91, [64, 128, -1, 64]);  view_91 = None
        permute_45: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_28: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        expand_17: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_28, [64, 16, 128, 64]);  mul_28 = None
        clone_29: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_99: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [1024, 128, 64]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_93: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [8192, 1024])
        permute_46: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg73_1, view_93, permute_46);  arg73_1 = view_93 = permute_46 = None
        view_94: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [64, 128, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_97: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [64, 128, -1, 64]);  view_94 = None
        permute_48: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_50: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        mul_29: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_18: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_29, [64, 16, 64, 128]);  mul_29 = None
        clone_30: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_100: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [1024, 64, 128]);  clone_30 = None
        bmm_8: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
        view_101: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [64, 16, 128, 128]);  bmm_8 = None
        eq_4: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_101, -inf)
        logical_not_8: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_9: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_128: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None
        amax_4: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_128, [-1], True)
        sub_13: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_128, amax_4);  convert_element_type_128 = amax_4 = None
        exp_4: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_129: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        where_9: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_9, full_default_9, convert_element_type_129);  logical_not_9 = full_default_9 = convert_element_type_129 = None
        expand_19: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_9, [64, 16, 128, 128]);  where_9 = None
        view_102: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_19, [1024, 128, 128]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_95: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [8192, 1024]);  convert_element_type_116 = None
        permute_47: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg75_1, view_95, permute_47);  arg75_1 = view_95 = permute_47 = None
        view_96: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [64, 128, 1024]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_98: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_96, [64, 128, -1, 64]);  view_96 = None
        permute_49: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_20: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_49, [64, 16, 128, 64]);  permute_49 = None
        clone_31: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_103: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [1024, 128, 64]);  clone_31 = None
        bmm_9: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
        view_104: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [64, 16, 128, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_32: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [64, 128, -1]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_106: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [8192, 1024]);  view_105 = None
        permute_52: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg77_1, view_106, permute_52);  arg77_1 = view_106 = permute_52 = None
        view_107: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [64, 128, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_36: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_32, view_107);  add_32 = view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_135: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_135, [2], correction = 0, keepdim = True)
        getitem_18: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_135, getitem_19);  convert_element_type_135 = getitem_19 = None
        add_37: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_30: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_31: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, arg78_1);  mul_30 = arg78_1 = None
        add_38: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, arg79_1);  mul_31 = arg79_1 = None
        convert_element_type_136: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_108: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_136, [8192, 1024]);  convert_element_type_136 = None
        permute_53: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg81_1, view_108, permute_53);  arg81_1 = view_108 = permute_53 = None
        view_109: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [64, 128, 4096]);  addmm_28 = None
        relu_4: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_109);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_110: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_4, [8192, 4096]);  relu_4 = None
        permute_54: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg83_1, view_110, permute_54);  arg83_1 = view_110 = permute_54 = None
        view_111: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [64, 128, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_39: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_36, view_111);  add_36 = view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_143: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_143, [2], correction = 0, keepdim = True)
        getitem_20: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_10: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_10 = None
        scalar_tensor_11: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_11 = None
        full_default_10: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_15: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_143, getitem_21);  convert_element_type_143 = getitem_21 = None
        add_40: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_32: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_33: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, arg84_1);  mul_32 = arg84_1 = None
        add_41: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, arg85_1);  mul_33 = arg85_1 = None
        convert_element_type_144: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_112: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_144, [8192, 1024])
        permute_55: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg87_1, view_112, permute_55);  arg87_1 = view_112 = permute_55 = None
        view_113: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [64, 128, 1024]);  addmm_30 = None
        view_114: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [64, 128, -1, 64]);  view_113 = None
        permute_56: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_34: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_56, 0.3535533905932738);  permute_56 = None
        expand_21: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_34, [64, 16, 128, 64]);  mul_34 = None
        clone_36: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_121: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [1024, 128, 64]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_115: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_144, [8192, 1024])
        permute_57: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg89_1, view_115, permute_57);  arg89_1 = view_115 = permute_57 = None
        view_116: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [64, 128, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_119: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_116, [64, 128, -1, 64]);  view_116 = None
        permute_59: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_61: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        mul_35: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_61, 0.3535533905932738);  permute_61 = None
        expand_22: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_35, [64, 16, 64, 128]);  mul_35 = None
        clone_37: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_122: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [1024, 64, 128]);  clone_37 = None
        bmm_10: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_121, view_122);  view_121 = view_122 = None
        view_123: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [64, 16, 128, 128]);  bmm_10 = None
        eq_5: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_123, -inf)
        logical_not_10: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_11: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_156: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_123, torch.float32);  view_123 = None
        amax_5: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_156, [-1], True)
        sub_16: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_156, amax_5);  convert_element_type_156 = amax_5 = None
        exp_5: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_157: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        where_11: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_11, full_default_11, convert_element_type_157);  logical_not_11 = full_default_11 = convert_element_type_157 = None
        expand_23: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_11, [64, 16, 128, 128]);  where_11 = None
        view_124: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_23, [1024, 128, 128]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_117: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_144, [8192, 1024]);  convert_element_type_144 = None
        permute_58: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg91_1, view_117, permute_58);  arg91_1 = view_117 = permute_58 = None
        view_118: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [64, 128, 1024]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_120: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_118, [64, 128, -1, 64]);  view_118 = None
        permute_60: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_24: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_60, [64, 16, 128, 64]);  permute_60 = None
        clone_38: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_125: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [1024, 128, 64]);  clone_38 = None
        bmm_11: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
        view_126: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [64, 16, 128, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_39: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_127: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [64, 128, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_128: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_127, [8192, 1024]);  view_127 = None
        permute_63: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg93_1, view_128, permute_63);  arg93_1 = view_128 = permute_63 = None
        view_129: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [64, 128, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_43: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_39, view_129);  add_39 = view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_163: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_163, [2], correction = 0, keepdim = True)
        getitem_22: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, getitem_23);  convert_element_type_163 = getitem_23 = None
        add_44: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_36: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_37: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg94_1);  mul_36 = arg94_1 = None
        add_45: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg95_1);  mul_37 = arg95_1 = None
        convert_element_type_164: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_130: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_164, [8192, 1024]);  convert_element_type_164 = None
        permute_64: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg97_1, view_130, permute_64);  arg97_1 = view_130 = permute_64 = None
        view_131: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [64, 128, 4096]);  addmm_34 = None
        relu_5: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_131);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_132: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_5, [8192, 4096]);  relu_5 = None
        permute_65: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg99_1, view_132, permute_65);  arg99_1 = view_132 = permute_65 = None
        view_133: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [64, 128, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_46: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_43, view_133);  add_43 = view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_171: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_171, [2], correction = 0, keepdim = True)
        getitem_24: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_12: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_12 = None
        scalar_tensor_13: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_13 = None
        full_default_12: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_18: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_171, getitem_25);  convert_element_type_171 = getitem_25 = None
        add_47: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_38: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_39: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, arg100_1);  mul_38 = arg100_1 = None
        add_48: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, arg101_1);  mul_39 = arg101_1 = None
        convert_element_type_172: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_134: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_172, [8192, 1024])
        permute_66: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg103_1, view_134, permute_66);  arg103_1 = view_134 = permute_66 = None
        view_135: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [64, 128, 1024]);  addmm_36 = None
        view_136: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_135, [64, 128, -1, 64]);  view_135 = None
        permute_67: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_40: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_67, 0.3535533905932738);  permute_67 = None
        expand_25: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_40, [64, 16, 128, 64]);  mul_40 = None
        clone_43: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_143: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [1024, 128, 64]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_137: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_172, [8192, 1024])
        permute_68: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_37: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg105_1, view_137, permute_68);  arg105_1 = view_137 = permute_68 = None
        view_138: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [64, 128, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_141: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_138, [64, 128, -1, 64]);  view_138 = None
        permute_70: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_72: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        mul_41: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_72, 0.3535533905932738);  permute_72 = None
        expand_26: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_41, [64, 16, 64, 128]);  mul_41 = None
        clone_44: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_144: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [1024, 64, 128]);  clone_44 = None
        bmm_12: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_143, view_144);  view_143 = view_144 = None
        view_145: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [64, 16, 128, 128]);  bmm_12 = None
        eq_6: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_145, -inf)
        logical_not_12: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_default_13: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_184: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_145, torch.float32);  view_145 = None
        amax_6: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_184, [-1], True)
        sub_19: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_184, amax_6);  convert_element_type_184 = amax_6 = None
        exp_6: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_185: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        where_13: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_13, full_default_13, convert_element_type_185);  logical_not_13 = full_default_13 = convert_element_type_185 = None
        expand_27: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_13, [64, 16, 128, 128]);  where_13 = None
        view_146: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_27, [1024, 128, 128]);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_139: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_172, [8192, 1024]);  convert_element_type_172 = None
        permute_69: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_38: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg107_1, view_139, permute_69);  arg107_1 = view_139 = permute_69 = None
        view_140: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [64, 128, 1024]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_142: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_140, [64, 128, -1, 64]);  view_140 = None
        permute_71: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_28: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_71, [64, 16, 128, 64]);  permute_71 = None
        clone_45: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_147: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [1024, 128, 64]);  clone_45 = None
        bmm_13: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_146, view_147);  view_146 = view_147 = None
        view_148: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [64, 16, 128, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_46: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_149: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [64, 128, -1]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_150: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_149, [8192, 1024]);  view_149 = None
        permute_74: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_39: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg109_1, view_150, permute_74);  arg109_1 = view_150 = permute_74 = None
        view_151: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [64, 128, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_50: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_46, view_151);  add_46 = view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_191: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_191, [2], correction = 0, keepdim = True)
        getitem_26: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_191, getitem_27);  convert_element_type_191 = getitem_27 = None
        add_51: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_42: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_43: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, arg110_1);  mul_42 = arg110_1 = None
        add_52: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, arg111_1);  mul_43 = arg111_1 = None
        convert_element_type_192: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_152: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_192, [8192, 1024]);  convert_element_type_192 = None
        permute_75: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_40: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg113_1, view_152, permute_75);  arg113_1 = view_152 = permute_75 = None
        view_153: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [64, 128, 4096]);  addmm_40 = None
        relu_6: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_153);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_154: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_6, [8192, 4096]);  relu_6 = None
        permute_76: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_41: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg115_1, view_154, permute_76);  arg115_1 = view_154 = permute_76 = None
        view_155: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [64, 128, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_53: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_50, view_155);  add_50 = view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_199: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_199, [2], correction = 0, keepdim = True)
        getitem_28: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_14: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_14 = None
        scalar_tensor_15: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_15 = None
        full_default_14: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_21: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_199, getitem_29);  convert_element_type_199 = getitem_29 = None
        add_54: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_44: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_45: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, arg116_1);  mul_44 = arg116_1 = None
        add_55: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, arg117_1);  mul_45 = arg117_1 = None
        convert_element_type_200: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_156: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_200, [8192, 1024])
        permute_77: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_42: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg119_1, view_156, permute_77);  arg119_1 = view_156 = permute_77 = None
        view_157: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [64, 128, 1024]);  addmm_42 = None
        view_158: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_157, [64, 128, -1, 64]);  view_157 = None
        permute_78: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_46: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_78, 0.3535533905932738);  permute_78 = None
        expand_29: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_46, [64, 16, 128, 64]);  mul_46 = None
        clone_50: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_165: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [1024, 128, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_159: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_200, [8192, 1024])
        permute_79: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_43: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg121_1, view_159, permute_79);  arg121_1 = view_159 = permute_79 = None
        view_160: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [64, 128, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_163: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_160, [64, 128, -1, 64]);  view_160 = None
        permute_81: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_83: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        mul_47: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_83, 0.3535533905932738);  permute_83 = None
        expand_30: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_47, [64, 16, 64, 128]);  mul_47 = None
        clone_51: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_166: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [1024, 64, 128]);  clone_51 = None
        bmm_14: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_165, view_166);  view_165 = view_166 = None
        view_167: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [64, 16, 128, 128]);  bmm_14 = None
        eq_7: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_167, -inf)
        logical_not_14: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_default_15: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_212: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_167, torch.float32);  view_167 = None
        amax_7: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_212, [-1], True)
        sub_22: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_212, amax_7);  convert_element_type_212 = amax_7 = None
        exp_7: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_213: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        where_15: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_15, full_default_15, convert_element_type_213);  logical_not_15 = full_default_15 = convert_element_type_213 = None
        expand_31: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_15, [64, 16, 128, 128]);  where_15 = None
        view_168: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_31, [1024, 128, 128]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_161: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_200, [8192, 1024]);  convert_element_type_200 = None
        permute_80: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_44: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg123_1, view_161, permute_80);  arg123_1 = view_161 = permute_80 = None
        view_162: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [64, 128, 1024]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_164: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_162, [64, 128, -1, 64]);  view_162 = None
        permute_82: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_32: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_82, [64, 16, 128, 64]);  permute_82 = None
        clone_52: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_169: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [1024, 128, 64]);  clone_52 = None
        bmm_15: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
        view_170: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [64, 16, 128, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_53: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_171: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [64, 128, -1]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_172: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_171, [8192, 1024]);  view_171 = None
        permute_85: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_45: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg125_1, view_172, permute_85);  arg125_1 = view_172 = permute_85 = None
        view_173: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [64, 128, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_57: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_53, view_173);  add_53 = view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_219: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_219, [2], correction = 0, keepdim = True)
        getitem_30: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_219, getitem_31);  convert_element_type_219 = getitem_31 = None
        add_58: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_48: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_49: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, arg126_1);  mul_48 = arg126_1 = None
        add_59: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, arg127_1);  mul_49 = arg127_1 = None
        convert_element_type_220: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_174: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_220, [8192, 1024]);  convert_element_type_220 = None
        permute_86: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_46: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg129_1, view_174, permute_86);  arg129_1 = view_174 = permute_86 = None
        view_175: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [64, 128, 4096]);  addmm_46 = None
        relu_7: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_175);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_176: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_7, [8192, 4096]);  relu_7 = None
        permute_87: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_47: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg131_1, view_176, permute_87);  arg131_1 = view_176 = permute_87 = None
        view_177: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [64, 128, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_60: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, view_177);  add_57 = view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_227: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_60, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_227, [2], correction = 0, keepdim = True)
        getitem_32: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_16: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_16 = None
        scalar_tensor_17: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_17 = None
        full_default_16: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_24: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_227, getitem_33);  convert_element_type_227 = getitem_33 = None
        add_61: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_50: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_51: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, arg132_1);  mul_50 = arg132_1 = None
        add_62: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, arg133_1);  mul_51 = arg133_1 = None
        convert_element_type_228: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_178: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_228, [8192, 1024])
        permute_88: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_48: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg135_1, view_178, permute_88);  arg135_1 = view_178 = permute_88 = None
        view_179: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [64, 128, 1024]);  addmm_48 = None
        view_180: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_179, [64, 128, -1, 64]);  view_179 = None
        permute_89: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_52: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_89, 0.3535533905932738);  permute_89 = None
        expand_33: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_52, [64, 16, 128, 64]);  mul_52 = None
        clone_57: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_187: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [1024, 128, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_181: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_228, [8192, 1024])
        permute_90: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_49: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg137_1, view_181, permute_90);  arg137_1 = view_181 = permute_90 = None
        view_182: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [64, 128, 1024]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_185: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_182, [64, 128, -1, 64]);  view_182 = None
        permute_92: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_94: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
        mul_53: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_94, 0.3535533905932738);  permute_94 = None
        expand_34: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_53, [64, 16, 64, 128]);  mul_53 = None
        clone_58: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_188: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [1024, 64, 128]);  clone_58 = None
        bmm_16: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
        view_189: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [64, 16, 128, 128]);  bmm_16 = None
        eq_8: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_189, -inf)
        logical_not_16: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_default_17: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_240: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_189, torch.float32);  view_189 = None
        amax_8: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_240, [-1], True)
        sub_25: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, amax_8);  convert_element_type_240 = amax_8 = None
        exp_8: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_241: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        where_17: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_17, full_default_17, convert_element_type_241);  logical_not_17 = full_default_17 = convert_element_type_241 = None
        expand_35: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_17, [64, 16, 128, 128]);  where_17 = None
        view_190: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_35, [1024, 128, 128]);  expand_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_183: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_228, [8192, 1024]);  convert_element_type_228 = None
        permute_91: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_50: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg139_1, view_183, permute_91);  arg139_1 = view_183 = permute_91 = None
        view_184: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [64, 128, 1024]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_186: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_184, [64, 128, -1, 64]);  view_184 = None
        permute_93: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_36: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_93, [64, 16, 128, 64]);  permute_93 = None
        clone_59: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_191: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [1024, 128, 64]);  clone_59 = None
        bmm_17: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
        view_192: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [64, 16, 128, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_60: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_193: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [64, 128, -1]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_194: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_193, [8192, 1024]);  view_193 = None
        permute_96: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_51: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg141_1, view_194, permute_96);  arg141_1 = view_194 = permute_96 = None
        view_195: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [64, 128, 1024]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_64: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_60, view_195);  add_60 = view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_247: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_247, [2], correction = 0, keepdim = True)
        getitem_34: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_247, getitem_35);  convert_element_type_247 = getitem_35 = None
        add_65: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_54: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_55: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, arg142_1);  mul_54 = arg142_1 = None
        add_66: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, arg143_1);  mul_55 = arg143_1 = None
        convert_element_type_248: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_196: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_248, [8192, 1024]);  convert_element_type_248 = None
        permute_97: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_52: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg145_1, view_196, permute_97);  arg145_1 = view_196 = permute_97 = None
        view_197: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [64, 128, 4096]);  addmm_52 = None
        relu_8: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_197);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_198: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_8, [8192, 4096]);  relu_8 = None
        permute_98: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_53: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg147_1, view_198, permute_98);  arg147_1 = view_198 = permute_98 = None
        view_199: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [64, 128, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_67: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_64, view_199);  add_64 = view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_255: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_255, [2], correction = 0, keepdim = True)
        getitem_36: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_18: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_18 = None
        scalar_tensor_19: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_19 = None
        full_default_18: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_27: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_255, getitem_37);  convert_element_type_255 = getitem_37 = None
        add_68: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_56: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_57: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, arg148_1);  mul_56 = arg148_1 = None
        add_69: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, arg149_1);  mul_57 = arg149_1 = None
        convert_element_type_256: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_200: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_256, [8192, 1024])
        permute_99: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_54: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg151_1, view_200, permute_99);  arg151_1 = view_200 = permute_99 = None
        view_201: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [64, 128, 1024]);  addmm_54 = None
        view_202: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_201, [64, 128, -1, 64]);  view_201 = None
        permute_100: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_58: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_100, 0.3535533905932738);  permute_100 = None
        expand_37: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_58, [64, 16, 128, 64]);  mul_58 = None
        clone_64: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_209: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [1024, 128, 64]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_203: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_256, [8192, 1024])
        permute_101: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_55: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg153_1, view_203, permute_101);  arg153_1 = view_203 = permute_101 = None
        view_204: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [64, 128, 1024]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_207: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_204, [64, 128, -1, 64]);  view_204 = None
        permute_103: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_105: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
        mul_59: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_105, 0.3535533905932738);  permute_105 = None
        expand_38: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_59, [64, 16, 64, 128]);  mul_59 = None
        clone_65: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_210: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1024, 64, 128]);  clone_65 = None
        bmm_18: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_209, view_210);  view_209 = view_210 = None
        view_211: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [64, 16, 128, 128]);  bmm_18 = None
        eq_9: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_211, -inf)
        logical_not_18: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_default_19: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_268: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_211, torch.float32);  view_211 = None
        amax_9: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_268, [-1], True)
        sub_28: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_268, amax_9);  convert_element_type_268 = amax_9 = None
        exp_9: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_269: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        where_19: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_19, full_default_19, convert_element_type_269);  logical_not_19 = full_default_19 = convert_element_type_269 = None
        expand_39: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_19, [64, 16, 128, 128]);  where_19 = None
        view_212: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_39, [1024, 128, 128]);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_205: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_256, [8192, 1024]);  convert_element_type_256 = None
        permute_102: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_56: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg155_1, view_205, permute_102);  arg155_1 = view_205 = permute_102 = None
        view_206: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [64, 128, 1024]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_208: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_206, [64, 128, -1, 64]);  view_206 = None
        permute_104: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_40: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_104, [64, 16, 128, 64]);  permute_104 = None
        clone_66: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_213: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [1024, 128, 64]);  clone_66 = None
        bmm_19: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_212, view_213);  view_212 = view_213 = None
        view_214: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [64, 16, 128, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_67: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [64, 128, -1]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_216: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_215, [8192, 1024]);  view_215 = None
        permute_107: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_57: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg157_1, view_216, permute_107);  arg157_1 = view_216 = permute_107 = None
        view_217: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [64, 128, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_71: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_67, view_217);  add_67 = view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_275: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_275, [2], correction = 0, keepdim = True)
        getitem_38: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_275, getitem_39);  convert_element_type_275 = getitem_39 = None
        add_72: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_60: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_61: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, arg158_1);  mul_60 = arg158_1 = None
        add_73: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, arg159_1);  mul_61 = arg159_1 = None
        convert_element_type_276: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_218: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_276, [8192, 1024]);  convert_element_type_276 = None
        permute_108: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_58: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg161_1, view_218, permute_108);  arg161_1 = view_218 = permute_108 = None
        view_219: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [64, 128, 4096]);  addmm_58 = None
        relu_9: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_219);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_220: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_9, [8192, 4096]);  relu_9 = None
        permute_109: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_59: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg163_1, view_220, permute_109);  arg163_1 = view_220 = permute_109 = None
        view_221: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [64, 128, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_74: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_71, view_221);  add_71 = view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_283: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_283, [2], correction = 0, keepdim = True)
        getitem_40: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_20: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_20 = None
        scalar_tensor_21: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_21 = None
        full_default_20: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_30: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_283, getitem_41);  convert_element_type_283 = getitem_41 = None
        add_75: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_62: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_63: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, arg164_1);  mul_62 = arg164_1 = None
        add_76: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, arg165_1);  mul_63 = arg165_1 = None
        convert_element_type_284: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_222: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_284, [8192, 1024])
        permute_110: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_60: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg167_1, view_222, permute_110);  arg167_1 = view_222 = permute_110 = None
        view_223: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [64, 128, 1024]);  addmm_60 = None
        view_224: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_223, [64, 128, -1, 64]);  view_223 = None
        permute_111: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_64: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_111, 0.3535533905932738);  permute_111 = None
        expand_41: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_64, [64, 16, 128, 64]);  mul_64 = None
        clone_71: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_231: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [1024, 128, 64]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_225: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_284, [8192, 1024])
        permute_112: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_61: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg169_1, view_225, permute_112);  arg169_1 = view_225 = permute_112 = None
        view_226: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [64, 128, 1024]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_229: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_226, [64, 128, -1, 64]);  view_226 = None
        permute_114: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_116: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
        mul_65: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_116, 0.3535533905932738);  permute_116 = None
        expand_42: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_65, [64, 16, 64, 128]);  mul_65 = None
        clone_72: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_232: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [1024, 64, 128]);  clone_72 = None
        bmm_20: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
        view_233: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [64, 16, 128, 128]);  bmm_20 = None
        eq_10: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_233, -inf)
        logical_not_20: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_default_21: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_296: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_233, torch.float32);  view_233 = None
        amax_10: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_296, [-1], True)
        sub_31: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_296, amax_10);  convert_element_type_296 = amax_10 = None
        exp_10: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_297: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        where_21: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_21, full_default_21, convert_element_type_297);  logical_not_21 = full_default_21 = convert_element_type_297 = None
        expand_43: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_21, [64, 16, 128, 128]);  where_21 = None
        view_234: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_43, [1024, 128, 128]);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_227: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_284, [8192, 1024]);  convert_element_type_284 = None
        permute_113: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_62: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg171_1, view_227, permute_113);  arg171_1 = view_227 = permute_113 = None
        view_228: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [64, 128, 1024]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_230: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_228, [64, 128, -1, 64]);  view_228 = None
        permute_115: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_44: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_115, [64, 16, 128, 64]);  permute_115 = None
        clone_73: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_235: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [1024, 128, 64]);  clone_73 = None
        bmm_21: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_234, view_235);  view_234 = view_235 = None
        view_236: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [64, 16, 128, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_74: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_237: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [64, 128, -1]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_238: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_237, [8192, 1024]);  view_237 = None
        permute_118: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_63: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg173_1, view_238, permute_118);  arg173_1 = view_238 = permute_118 = None
        view_239: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [64, 128, 1024]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_78: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_74, view_239);  add_74 = view_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_303: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_303, [2], correction = 0, keepdim = True)
        getitem_42: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_303, getitem_43);  convert_element_type_303 = getitem_43 = None
        add_79: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_66: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_67: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg174_1);  mul_66 = arg174_1 = None
        add_80: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg175_1);  mul_67 = arg175_1 = None
        convert_element_type_304: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_240: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_304, [8192, 1024]);  convert_element_type_304 = None
        permute_119: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_64: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg177_1, view_240, permute_119);  arg177_1 = view_240 = permute_119 = None
        view_241: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [64, 128, 4096]);  addmm_64 = None
        relu_10: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_241);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_242: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_10, [8192, 4096]);  relu_10 = None
        permute_120: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_65: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg179_1, view_242, permute_120);  arg179_1 = view_242 = permute_120 = None
        view_243: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [64, 128, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_81: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_78, view_243);  add_78 = view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_311: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_311, [2], correction = 0, keepdim = True)
        getitem_44: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_22: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_22 = None
        scalar_tensor_23: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_23 = None
        full_default_22: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_33: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_311, getitem_45);  convert_element_type_311 = getitem_45 = None
        add_82: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_68: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_69: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, arg180_1);  mul_68 = arg180_1 = None
        add_83: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, arg181_1);  mul_69 = arg181_1 = None
        convert_element_type_312: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_244: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_312, [8192, 1024])
        permute_121: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_66: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg183_1, view_244, permute_121);  arg183_1 = view_244 = permute_121 = None
        view_245: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [64, 128, 1024]);  addmm_66 = None
        view_246: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [64, 128, -1, 64]);  view_245 = None
        permute_122: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_70: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_122, 0.3535533905932738);  permute_122 = None
        expand_45: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_70, [64, 16, 128, 64]);  mul_70 = None
        clone_78: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_253: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [1024, 128, 64]);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_247: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_312, [8192, 1024])
        permute_123: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_67: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg185_1, view_247, permute_123);  arg185_1 = view_247 = permute_123 = None
        view_248: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [64, 128, 1024]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_251: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_248, [64, 128, -1, 64]);  view_248 = None
        permute_125: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_127: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        mul_71: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_127, 0.3535533905932738);  permute_127 = None
        expand_46: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_71, [64, 16, 64, 128]);  mul_71 = None
        clone_79: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_254: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [1024, 64, 128]);  clone_79 = None
        bmm_22: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_253, view_254);  view_253 = view_254 = None
        view_255: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [64, 16, 128, 128]);  bmm_22 = None
        eq_11: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_255, -inf)
        logical_not_22: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_default_23: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_324: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_255, torch.float32);  view_255 = None
        amax_11: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_324, [-1], True)
        sub_34: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_324, amax_11);  convert_element_type_324 = amax_11 = None
        exp_11: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_325: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        where_23: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_23, full_default_23, convert_element_type_325);  logical_not_23 = full_default_23 = convert_element_type_325 = None
        expand_47: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_23, [64, 16, 128, 128]);  where_23 = None
        view_256: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_47, [1024, 128, 128]);  expand_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_249: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_312, [8192, 1024]);  convert_element_type_312 = None
        permute_124: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_68: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg187_1, view_249, permute_124);  arg187_1 = view_249 = permute_124 = None
        view_250: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [64, 128, 1024]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_252: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_250, [64, 128, -1, 64]);  view_250 = None
        permute_126: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_48: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_126, [64, 16, 128, 64]);  permute_126 = None
        clone_80: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_257: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [1024, 128, 64]);  clone_80 = None
        bmm_23: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_256, view_257);  view_256 = view_257 = None
        view_258: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [64, 16, 128, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_81: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_259: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [64, 128, -1]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_260: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_259, [8192, 1024]);  view_259 = None
        permute_129: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_69: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg189_1, view_260, permute_129);  arg189_1 = view_260 = permute_129 = None
        view_261: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [64, 128, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:369 in forward, code: hidden_states = residual + hidden_states
        add_85: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_81, view_261);  add_81 = view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_331: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_331, [2], correction = 0, keepdim = True)
        getitem_46: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, getitem_47);  convert_element_type_331 = getitem_47 = None
        add_86: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_72: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_73: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, arg190_1);  mul_72 = arg190_1 = None
        add_87: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, arg191_1);  mul_73 = arg191_1 = None
        convert_element_type_332: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_262: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_332, [8192, 1024]);  convert_element_type_332 = None
        permute_130: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_70: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg193_1, view_262, permute_130);  arg193_1 = view_262 = permute_130 = None
        view_263: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [64, 128, 4096]);  addmm_70 = None
        relu_11: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_263);  view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        view_264: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_11, [8192, 4096]);  relu_11 = None
        permute_131: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_71: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg195_1, view_264, permute_131);  arg195_1 = view_264 = permute_131 = None
        view_265: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [64, 128, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_88: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_85, view_265);  add_85 = view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:586 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_339: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.float32);  add_88 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_339, [2], correction = 0, keepdim = True)
        getitem_48: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_10: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_93: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_10, 0);  iota_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_9: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_93, 0);  add_93 = None
        unsqueeze_10: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        unsqueeze_11: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_1: "b8[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_11, 0);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_50: "b8[64, 1, 128, 128][0, 128, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge_1, [64, -1, 128, 128]);  ge_1 = expand_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:77 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_1: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 1)
        mul_76: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding_1, 32.0);  embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:177 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne_1: "b8[64, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg1_1, 1);  arg1_1 = None
        convert_element_type_341: "i32[64, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(ne_1, torch.int32);  ne_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:178 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        cumsum_1: "i64[64, 128][128, 1]cuda:0" = torch.ops.aten.cumsum.default(convert_element_type_341, 1)
        convert_element_type_342: "i32[64, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cumsum_1, torch.int32);  cumsum_1 = None
        add_95: "i32[64, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_342, 0);  convert_element_type_342 = None
        mul_77: "i32[64, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_95, convert_element_type_341);  add_95 = convert_element_type_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:179 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_343: "i64[64, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_77, torch.int64);  mul_77 = None
        add_96: "i64[64, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_343, 1);  convert_element_type_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:144 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        view_266: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(add_96, [-1]);  add_96 = None
        index_1: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.index.Tensor(arg198_1, [view_266]);  arg198_1 = view_266 = None
        view_267: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(index_1, [64, 128, 1024]);  index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:687 in forward, code: hidden_states = inputs_embeds + positions
        add_97: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, view_267);  mul_76 = view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_344: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_344, [2], correction = 0, keepdim = True)
        getitem_50: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        sub_37: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_344, getitem_51);  convert_element_type_344 = getitem_51 = None
        add_98: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_78: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
        mul_79: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, arg199_1);  mul_78 = arg199_1 = None
        add_99: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, arg200_1);  mul_79 = arg200_1 = None
        convert_element_type_345: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_268: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_345, [8192, 1024])
        permute_132: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_72: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg202_1, view_268, permute_132);  arg202_1 = view_268 = permute_132 = None
        view_269: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [64, 128, 1024]);  addmm_72 = None
        view_270: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_269, [64, 128, -1, 64]);  view_269 = None
        permute_133: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_271: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_345, [8192, 1024])
        permute_134: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_73: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg204_1, view_271, permute_134);  arg204_1 = view_271 = permute_134 = None
        view_272: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [64, 128, 1024]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_275: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_272, [64, 128, -1, 64]);  view_272 = None
        permute_136: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_273: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_345, [8192, 1024]);  convert_element_type_345 = None
        permute_135: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_74: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg206_1, view_273, permute_135);  arg206_1 = view_273 = permute_135 = None
        view_274: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [64, 128, 1024]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_276: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_274, [64, 128, -1, 64]);  view_274 = None
        permute_137: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_276, [0, 2, 1, 3]);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_7: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_92: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_7, 0);  iota_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_6: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_92, 0);  add_92 = None
        unsqueeze_7: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_6: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_91: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_6, 0);  iota_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_3: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_91, 0);  add_91 = None
        unsqueeze_4: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_8, unsqueeze_5);  unsqueeze_8 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_49: "b8[64, 1, 128, 128][0, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(le, [64, -1, 128, 128]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_25: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_25, full_default_24);  full_default_25 = full_default_24 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_133, permute_136, permute_137, where_24, False, scale = 0.125);  permute_133 = permute_136 = permute_137 = where_24 = None
        getitem_52: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_138: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_277: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_138, [64, 128, -1]);  permute_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_278: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_277, [8192, 1024]);  view_277 = None
        permute_139: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_75: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg208_1, view_278, permute_139);  arg208_1 = view_278 = permute_139 = None
        view_279: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [64, 128, 1024]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_100: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, view_279);  add_97 = view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_358: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_358, [2], correction = 0, keepdim = True)
        getitem_61: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_26[0]
        getitem_62: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_26: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_26 = None
        scalar_tensor_27: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_27 = None
        full_default_26: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_38: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_358, getitem_62);  convert_element_type_358 = getitem_62 = None
        add_101: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_61, 1e-05);  getitem_61 = None
        rsqrt_26: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_80: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = rsqrt_26 = None
        mul_81: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, arg209_1);  mul_80 = arg209_1 = None
        add_102: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, arg210_1);  mul_81 = arg210_1 = None
        convert_element_type_359: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_280: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_359, [8192, 1024]);  convert_element_type_359 = None
        permute_140: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_76: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg212_1, view_280, permute_140);  arg212_1 = view_280 = permute_140 = None
        view_281: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [64, 128, 1024]);  addmm_76 = None
        view_282: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_281, [64, 128, -1, 64]);  view_281 = None
        permute_141: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_82: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_141, 0.3535533905932738);  permute_141 = None
        expand_51: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_82, [64, 16, 128, 64]);  mul_82 = None
        clone_87: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_289: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [1024, 128, 64]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:586 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_36: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_339, getitem_49);  convert_element_type_339 = getitem_49 = None
        add_89: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_74: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_75: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, arg196_1);  mul_74 = arg196_1 = None
        add_90: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_75, arg197_1);  mul_75 = arg197_1 = None
        convert_element_type_340: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_283: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_142: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_77: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg214_1, view_283, permute_142);  arg214_1 = view_283 = permute_142 = None
        view_284: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [64, 128, 1024]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_287: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_284, [64, 128, -1, 64]);  view_284 = None
        permute_144: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_146: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_144, [0, 1, 3, 2]);  permute_144 = None
        mul_83: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_146, 0.3535533905932738);  permute_146 = None
        expand_52: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_83, [64, 16, 64, 128]);  mul_83 = None
        clone_88: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_290: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [1024, 64, 128]);  clone_88 = None
        bmm_24: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_289, view_290);  view_289 = view_290 = None
        view_291: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [64, 16, 128, 128]);  bmm_24 = None
        eq_12: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_291, -inf)
        logical_not_24: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_12);  eq_12 = None
        any_13: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_24, -1, True);  logical_not_24 = None
        logical_not_25: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        full_default_27: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_371: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_291, torch.float32);  view_291 = None
        amax_12: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_371, [-1], True)
        sub_39: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_371, amax_12);  convert_element_type_371 = amax_12 = None
        exp_12: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_13: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_372: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        where_26: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_25, full_default_27, convert_element_type_372);  logical_not_25 = full_default_27 = convert_element_type_372 = None
        expand_53: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_26, [64, 16, 128, 128]);  where_26 = None
        view_292: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_53, [1024, 128, 128]);  expand_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_285: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_143: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_78: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg216_1, view_285, permute_143);  arg216_1 = view_285 = permute_143 = None
        view_286: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [64, 128, 1024]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_288: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_286, [64, 128, -1, 64]);  view_286 = None
        permute_145: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_54: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_145, [64, 16, 128, 64]);  permute_145 = None
        clone_89: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_293: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [1024, 128, 64]);  clone_89 = None
        bmm_25: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_292, view_293);  view_292 = view_293 = None
        view_294: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [64, 16, 128, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_147: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        clone_90: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_295: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [64, 128, -1]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_296: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [8192, 1024]);  view_295 = None
        permute_148: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_79: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg218_1, view_296, permute_148);  arg218_1 = view_296 = permute_148 = None
        view_297: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [64, 128, 1024]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_104: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_100, view_297);  add_100 = view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_378: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_378, [2], correction = 0, keepdim = True)
        getitem_63: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_27[0]
        getitem_64: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        sub_40: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_378, getitem_64);  convert_element_type_378 = getitem_64 = None
        add_105: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_27: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        mul_84: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_27);  sub_40 = rsqrt_27 = None
        mul_85: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, arg219_1);  mul_84 = arg219_1 = None
        add_106: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, arg220_1);  mul_85 = arg220_1 = None
        convert_element_type_379: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16);  add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_298: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_379, [8192, 1024]);  convert_element_type_379 = None
        permute_149: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_80: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg222_1, view_298, permute_149);  arg222_1 = view_298 = permute_149 = None
        view_299: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [64, 128, 4096]);  addmm_80 = None
        relu_12: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_299);  view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_300: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_12, [8192, 4096]);  relu_12 = None
        permute_150: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_81: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg224_1, view_300, permute_150);  arg224_1 = view_300 = permute_150 = None
        view_301: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [64, 128, 1024]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_107: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_104, view_301);  add_104 = view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_386: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_386, [2], correction = 0, keepdim = True)
        getitem_65: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_28[0]
        getitem_66: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        sub_41: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_386, getitem_66);  convert_element_type_386 = getitem_66 = None
        add_108: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_28: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_86: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = rsqrt_28 = None
        mul_87: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, arg225_1);  mul_86 = arg225_1 = None
        add_109: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, arg226_1);  mul_87 = arg226_1 = None
        convert_element_type_387: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_302: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_387, [8192, 1024])
        permute_151: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_82: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg228_1, view_302, permute_151);  arg228_1 = view_302 = permute_151 = None
        view_303: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [64, 128, 1024]);  addmm_82 = None
        view_304: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_303, [64, 128, -1, 64]);  view_303 = None
        permute_152: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_305: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_387, [8192, 1024])
        permute_153: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_83: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg230_1, view_305, permute_153);  arg230_1 = view_305 = permute_153 = None
        view_306: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [64, 128, 1024]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_309: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_306, [64, 128, -1, 64]);  view_306 = None
        permute_155: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_307: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_387, [8192, 1024]);  convert_element_type_387 = None
        permute_154: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_84: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg232_1, view_307, permute_154);  arg232_1 = view_307 = permute_154 = None
        view_308: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [64, 128, 1024]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_310: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_308, [64, 128, -1, 64]);  view_308 = None
        permute_156: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_29: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_29, full_default_28);  full_default_29 = full_default_28 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_152, permute_155, permute_156, where_27, False, scale = 0.125);  permute_152 = permute_155 = permute_156 = where_27 = None
        getitem_67: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_157: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_67, [0, 2, 1, 3]);  getitem_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_311: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_157, [64, 128, -1]);  permute_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_312: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_311, [8192, 1024]);  view_311 = None
        permute_158: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_85: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg234_1, view_312, permute_158);  arg234_1 = view_312 = permute_158 = None
        view_313: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_85, [64, 128, 1024]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_110: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_107, view_313);  add_107 = view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_400: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_110, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_400, [2], correction = 0, keepdim = True)
        getitem_76: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_29[0]
        getitem_77: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_30: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_30 = None
        scalar_tensor_31: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_31 = None
        full_default_30: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_42: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_400, getitem_77);  convert_element_type_400 = getitem_77 = None
        add_111: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_29: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_88: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_29);  sub_42 = rsqrt_29 = None
        mul_89: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, arg235_1);  mul_88 = arg235_1 = None
        add_112: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_89, arg236_1);  mul_89 = arg236_1 = None
        convert_element_type_401: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_314: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_401, [8192, 1024]);  convert_element_type_401 = None
        permute_159: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_86: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg238_1, view_314, permute_159);  arg238_1 = view_314 = permute_159 = None
        view_315: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [64, 128, 1024]);  addmm_86 = None
        view_316: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [64, 128, -1, 64]);  view_315 = None
        permute_160: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_90: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_160, 0.3535533905932738);  permute_160 = None
        expand_55: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_90, [64, 16, 128, 64]);  mul_90 = None
        clone_95: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_323: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [1024, 128, 64]);  clone_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_317: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_161: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_87: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg240_1, view_317, permute_161);  arg240_1 = view_317 = permute_161 = None
        view_318: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_87, [64, 128, 1024]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_321: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_318, [64, 128, -1, 64]);  view_318 = None
        permute_163: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_165: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_163, [0, 1, 3, 2]);  permute_163 = None
        mul_91: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_165, 0.3535533905932738);  permute_165 = None
        expand_56: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_91, [64, 16, 64, 128]);  mul_91 = None
        clone_96: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_324: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [1024, 64, 128]);  clone_96 = None
        bmm_26: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_323, view_324);  view_323 = view_324 = None
        view_325: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [64, 16, 128, 128]);  bmm_26 = None
        eq_13: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_325, -inf)
        logical_not_26: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_13);  eq_13 = None
        any_14: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_26, -1, True);  logical_not_26 = None
        logical_not_27: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        full_default_31: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_413: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_325, torch.float32);  view_325 = None
        amax_13: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_413, [-1], True)
        sub_43: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_413, amax_13);  convert_element_type_413 = amax_13 = None
        exp_13: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_14: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_414: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        where_29: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_27, full_default_31, convert_element_type_414);  logical_not_27 = full_default_31 = convert_element_type_414 = None
        expand_57: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_29, [64, 16, 128, 128]);  where_29 = None
        view_326: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_57, [1024, 128, 128]);  expand_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_319: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_162: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_88: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg242_1, view_319, permute_162);  arg242_1 = view_319 = permute_162 = None
        view_320: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [64, 128, 1024]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_322: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_320, [64, 128, -1, 64]);  view_320 = None
        permute_164: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_58: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_164, [64, 16, 128, 64]);  permute_164 = None
        clone_97: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_327: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [1024, 128, 64]);  clone_97 = None
        bmm_27: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_326, view_327);  view_326 = view_327 = None
        view_328: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [64, 16, 128, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_166: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        clone_98: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_329: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [64, 128, -1]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_330: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [8192, 1024]);  view_329 = None
        permute_167: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_89: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg244_1, view_330, permute_167);  arg244_1 = view_330 = permute_167 = None
        view_331: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_89, [64, 128, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_114: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, view_331);  add_110 = view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_420: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_420, [2], correction = 0, keepdim = True)
        getitem_78: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_30[0]
        getitem_79: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        sub_44: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_420, getitem_79);  convert_element_type_420 = getitem_79 = None
        add_115: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_30: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_92: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = rsqrt_30 = None
        mul_93: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, arg245_1);  mul_92 = arg245_1 = None
        add_116: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, arg246_1);  mul_93 = arg246_1 = None
        convert_element_type_421: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_332: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_421, [8192, 1024]);  convert_element_type_421 = None
        permute_168: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_90: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg248_1, view_332, permute_168);  arg248_1 = view_332 = permute_168 = None
        view_333: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [64, 128, 4096]);  addmm_90 = None
        relu_13: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_333);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_334: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_13, [8192, 4096]);  relu_13 = None
        permute_169: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_91: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg250_1, view_334, permute_169);  arg250_1 = view_334 = permute_169 = None
        view_335: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_91, [64, 128, 1024]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_117: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_114, view_335);  add_114 = view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_428: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_428, [2], correction = 0, keepdim = True)
        getitem_80: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_31[0]
        getitem_81: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        sub_45: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_428, getitem_81);  convert_element_type_428 = getitem_81 = None
        add_118: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_31: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_94: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_31);  sub_45 = rsqrt_31 = None
        mul_95: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, arg251_1);  mul_94 = arg251_1 = None
        add_119: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, arg252_1);  mul_95 = arg252_1 = None
        convert_element_type_429: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_336: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_429, [8192, 1024])
        permute_170: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_92: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg254_1, view_336, permute_170);  arg254_1 = view_336 = permute_170 = None
        view_337: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_92, [64, 128, 1024]);  addmm_92 = None
        view_338: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_337, [64, 128, -1, 64]);  view_337 = None
        permute_171: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_339: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_429, [8192, 1024])
        permute_172: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_93: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg256_1, view_339, permute_172);  arg256_1 = view_339 = permute_172 = None
        view_340: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_93, [64, 128, 1024]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_343: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_340, [64, 128, -1, 64]);  view_340 = None
        permute_174: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_341: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_429, [8192, 1024]);  convert_element_type_429 = None
        permute_173: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_94: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg258_1, view_341, permute_173);  arg258_1 = view_341 = permute_173 = None
        view_342: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [64, 128, 1024]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_344: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_342, [64, 128, -1, 64]);  view_342 = None
        permute_175: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_33: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_33, full_default_32);  full_default_33 = full_default_32 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_171, permute_174, permute_175, where_30, False, scale = 0.125);  permute_171 = permute_174 = permute_175 = where_30 = None
        getitem_82: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_176: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_345: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_176, [64, 128, -1]);  permute_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_346: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [8192, 1024]);  view_345 = None
        permute_177: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_95: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg260_1, view_346, permute_177);  arg260_1 = view_346 = permute_177 = None
        view_347: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_95, [64, 128, 1024]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_120: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, view_347);  add_117 = view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_442: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_442, [2], correction = 0, keepdim = True)
        getitem_91: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_32[0]
        getitem_92: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_34: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_34 = None
        scalar_tensor_35: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_35 = None
        full_default_34: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_46: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_442, getitem_92);  convert_element_type_442 = getitem_92 = None
        add_121: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_32: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        mul_96: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_32);  sub_46 = rsqrt_32 = None
        mul_97: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, arg261_1);  mul_96 = arg261_1 = None
        add_122: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, arg262_1);  mul_97 = arg262_1 = None
        convert_element_type_443: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16);  add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_348: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_443, [8192, 1024]);  convert_element_type_443 = None
        permute_178: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_96: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg264_1, view_348, permute_178);  arg264_1 = view_348 = permute_178 = None
        view_349: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_96, [64, 128, 1024]);  addmm_96 = None
        view_350: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_349, [64, 128, -1, 64]);  view_349 = None
        permute_179: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_98: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_179, 0.3535533905932738);  permute_179 = None
        expand_59: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_98, [64, 16, 128, 64]);  mul_98 = None
        clone_103: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_357: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [1024, 128, 64]);  clone_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_351: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_180: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_97: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg266_1, view_351, permute_180);  arg266_1 = view_351 = permute_180 = None
        view_352: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_97, [64, 128, 1024]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_355: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_352, [64, 128, -1, 64]);  view_352 = None
        permute_182: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_355, [0, 2, 1, 3]);  view_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_184: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_182, [0, 1, 3, 2]);  permute_182 = None
        mul_99: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_184, 0.3535533905932738);  permute_184 = None
        expand_60: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_99, [64, 16, 64, 128]);  mul_99 = None
        clone_104: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_358: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_104, [1024, 64, 128]);  clone_104 = None
        bmm_28: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_357, view_358);  view_357 = view_358 = None
        view_359: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [64, 16, 128, 128]);  bmm_28 = None
        eq_14: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_359, -inf)
        logical_not_28: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_14);  eq_14 = None
        any_15: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_28, -1, True);  logical_not_28 = None
        logical_not_29: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        full_default_35: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_455: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        amax_14: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_455, [-1], True)
        sub_47: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_455, amax_14);  convert_element_type_455 = amax_14 = None
        exp_14: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
        sum_15: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_456: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        where_32: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_29, full_default_35, convert_element_type_456);  logical_not_29 = full_default_35 = convert_element_type_456 = None
        expand_61: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_32, [64, 16, 128, 128]);  where_32 = None
        view_360: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_61, [1024, 128, 128]);  expand_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_353: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_181: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_98: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg268_1, view_353, permute_181);  arg268_1 = view_353 = permute_181 = None
        view_354: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_98, [64, 128, 1024]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_356: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_354, [64, 128, -1, 64]);  view_354 = None
        permute_183: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_62: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_183, [64, 16, 128, 64]);  permute_183 = None
        clone_105: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_361: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [1024, 128, 64]);  clone_105 = None
        bmm_29: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_360, view_361);  view_360 = view_361 = None
        view_362: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [64, 16, 128, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_185: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
        clone_106: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_363: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [64, 128, -1]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_364: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_363, [8192, 1024]);  view_363 = None
        permute_186: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_99: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg270_1, view_364, permute_186);  arg270_1 = view_364 = permute_186 = None
        view_365: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_99, [64, 128, 1024]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_124: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_120, view_365);  add_120 = view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_462: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_462, [2], correction = 0, keepdim = True)
        getitem_93: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_33[0]
        getitem_94: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        sub_48: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_462, getitem_94);  convert_element_type_462 = getitem_94 = None
        add_125: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_93, 1e-05);  getitem_93 = None
        rsqrt_33: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_100: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_33);  sub_48 = rsqrt_33 = None
        mul_101: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, arg271_1);  mul_100 = arg271_1 = None
        add_126: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, arg272_1);  mul_101 = arg272_1 = None
        convert_element_type_463: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_126, torch.bfloat16);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_366: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_463, [8192, 1024]);  convert_element_type_463 = None
        permute_187: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_100: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg274_1, view_366, permute_187);  arg274_1 = view_366 = permute_187 = None
        view_367: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_100, [64, 128, 4096]);  addmm_100 = None
        relu_14: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_367);  view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_368: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_14, [8192, 4096]);  relu_14 = None
        permute_188: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_101: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg276_1, view_368, permute_188);  arg276_1 = view_368 = permute_188 = None
        view_369: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_101, [64, 128, 1024]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_127: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_124, view_369);  add_124 = view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_470: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_470, [2], correction = 0, keepdim = True)
        getitem_95: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_34[0]
        getitem_96: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        sub_49: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_470, getitem_96);  convert_element_type_470 = getitem_96 = None
        add_128: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_95, 1e-05);  getitem_95 = None
        rsqrt_34: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_102: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_34);  sub_49 = rsqrt_34 = None
        mul_103: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, arg277_1);  mul_102 = arg277_1 = None
        add_129: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_103, arg278_1);  mul_103 = arg278_1 = None
        convert_element_type_471: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_370: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_471, [8192, 1024])
        permute_189: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_102: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg280_1, view_370, permute_189);  arg280_1 = view_370 = permute_189 = None
        view_371: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_102, [64, 128, 1024]);  addmm_102 = None
        view_372: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_371, [64, 128, -1, 64]);  view_371 = None
        permute_190: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_373: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_471, [8192, 1024])
        permute_191: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_103: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg282_1, view_373, permute_191);  arg282_1 = view_373 = permute_191 = None
        view_374: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_103, [64, 128, 1024]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_377: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_374, [64, 128, -1, 64]);  view_374 = None
        permute_193: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_375: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_471, [8192, 1024]);  convert_element_type_471 = None
        permute_192: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_104: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg284_1, view_375, permute_192);  arg284_1 = view_375 = permute_192 = None
        view_376: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_104, [64, 128, 1024]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_378: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_376, [64, 128, -1, 64]);  view_376 = None
        permute_194: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_37: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_33: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_37, full_default_36);  full_default_37 = full_default_36 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_190, permute_193, permute_194, where_33, False, scale = 0.125);  permute_190 = permute_193 = permute_194 = where_33 = None
        getitem_97: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_195: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_97, [0, 2, 1, 3]);  getitem_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_379: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_195, [64, 128, -1]);  permute_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_380: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_379, [8192, 1024]);  view_379 = None
        permute_196: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_105: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg286_1, view_380, permute_196);  arg286_1 = view_380 = permute_196 = None
        view_381: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_105, [64, 128, 1024]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_130: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_127, view_381);  add_127 = view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_484: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_130, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_484, [2], correction = 0, keepdim = True)
        getitem_106: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_35[0]
        getitem_107: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_38: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_38 = None
        scalar_tensor_39: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_39 = None
        full_default_38: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_50: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_484, getitem_107);  convert_element_type_484 = getitem_107 = None
        add_131: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
        rsqrt_35: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_104: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_35);  sub_50 = rsqrt_35 = None
        mul_105: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, arg287_1);  mul_104 = arg287_1 = None
        add_132: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_105, arg288_1);  mul_105 = arg288_1 = None
        convert_element_type_485: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.bfloat16);  add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_382: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_485, [8192, 1024]);  convert_element_type_485 = None
        permute_197: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_106: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg290_1, view_382, permute_197);  arg290_1 = view_382 = permute_197 = None
        view_383: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_106, [64, 128, 1024]);  addmm_106 = None
        view_384: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_383, [64, 128, -1, 64]);  view_383 = None
        permute_198: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_106: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_198, 0.3535533905932738);  permute_198 = None
        expand_63: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_106, [64, 16, 128, 64]);  mul_106 = None
        clone_111: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_391: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_111, [1024, 128, 64]);  clone_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_385: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_199: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_107: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg292_1, view_385, permute_199);  arg292_1 = view_385 = permute_199 = None
        view_386: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_107, [64, 128, 1024]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_389: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_386, [64, 128, -1, 64]);  view_386 = None
        permute_201: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_203: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_201, [0, 1, 3, 2]);  permute_201 = None
        mul_107: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_203, 0.3535533905932738);  permute_203 = None
        expand_64: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_107, [64, 16, 64, 128]);  mul_107 = None
        clone_112: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_392: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_112, [1024, 64, 128]);  clone_112 = None
        bmm_30: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_391, view_392);  view_391 = view_392 = None
        view_393: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [64, 16, 128, 128]);  bmm_30 = None
        eq_15: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_393, -inf)
        logical_not_30: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_15);  eq_15 = None
        any_16: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_30, -1, True);  logical_not_30 = None
        logical_not_31: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        full_default_39: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_497: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        amax_15: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_497, [-1], True)
        sub_51: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_497, amax_15);  convert_element_type_497 = amax_15 = None
        exp_15: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        sum_16: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_498: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        where_35: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_31, full_default_39, convert_element_type_498);  logical_not_31 = full_default_39 = convert_element_type_498 = None
        expand_65: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_35, [64, 16, 128, 128]);  where_35 = None
        view_394: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_65, [1024, 128, 128]);  expand_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_387: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_200: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_108: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg294_1, view_387, permute_200);  arg294_1 = view_387 = permute_200 = None
        view_388: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_108, [64, 128, 1024]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_390: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_388, [64, 128, -1, 64]);  view_388 = None
        permute_202: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_66: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_202, [64, 16, 128, 64]);  permute_202 = None
        clone_113: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_395: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [1024, 128, 64]);  clone_113 = None
        bmm_31: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_394, view_395);  view_394 = view_395 = None
        view_396: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [64, 16, 128, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_204: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None
        clone_114: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_397: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [64, 128, -1]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_398: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [8192, 1024]);  view_397 = None
        permute_205: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_109: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg296_1, view_398, permute_205);  arg296_1 = view_398 = permute_205 = None
        view_399: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_109, [64, 128, 1024]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_134: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, view_399);  add_130 = view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_504: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_504, [2], correction = 0, keepdim = True)
        getitem_108: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_36[0]
        getitem_109: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        sub_52: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_504, getitem_109);  convert_element_type_504 = getitem_109 = None
        add_135: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_36: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_108: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_36);  sub_52 = rsqrt_36 = None
        mul_109: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, arg297_1);  mul_108 = arg297_1 = None
        add_136: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, arg298_1);  mul_109 = arg298_1 = None
        convert_element_type_505: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_400: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [8192, 1024]);  convert_element_type_505 = None
        permute_206: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_110: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg300_1, view_400, permute_206);  arg300_1 = view_400 = permute_206 = None
        view_401: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_110, [64, 128, 4096]);  addmm_110 = None
        relu_15: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_401);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_402: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_15, [8192, 4096]);  relu_15 = None
        permute_207: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_111: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg302_1, view_402, permute_207);  arg302_1 = view_402 = permute_207 = None
        view_403: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_111, [64, 128, 1024]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_137: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, view_403);  add_134 = view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_512: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_512, [2], correction = 0, keepdim = True)
        getitem_110: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_37[0]
        getitem_111: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        sub_53: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_512, getitem_111);  convert_element_type_512 = getitem_111 = None
        add_138: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_37: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_110: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_37);  sub_53 = rsqrt_37 = None
        mul_111: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, arg303_1);  mul_110 = arg303_1 = None
        add_139: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, arg304_1);  mul_111 = arg304_1 = None
        convert_element_type_513: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_404: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_513, [8192, 1024])
        permute_208: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_112: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg306_1, view_404, permute_208);  arg306_1 = view_404 = permute_208 = None
        view_405: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_112, [64, 128, 1024]);  addmm_112 = None
        view_406: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_405, [64, 128, -1, 64]);  view_405 = None
        permute_209: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_407: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_513, [8192, 1024])
        permute_210: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_113: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg308_1, view_407, permute_210);  arg308_1 = view_407 = permute_210 = None
        view_408: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_113, [64, 128, 1024]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_411: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_408, [64, 128, -1, 64]);  view_408 = None
        permute_212: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_409: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_513, [8192, 1024]);  convert_element_type_513 = None
        permute_211: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_114: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg310_1, view_409, permute_211);  arg310_1 = view_409 = permute_211 = None
        view_410: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_114, [64, 128, 1024]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_412: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_410, [64, 128, -1, 64]);  view_410 = None
        permute_213: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_41: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_41, full_default_40);  full_default_41 = full_default_40 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_209, permute_212, permute_213, where_36, False, scale = 0.125);  permute_209 = permute_212 = permute_213 = where_36 = None
        getitem_112: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_214: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_413: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_214, [64, 128, -1]);  permute_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_414: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_413, [8192, 1024]);  view_413 = None
        permute_215: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_115: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg312_1, view_414, permute_215);  arg312_1 = view_414 = permute_215 = None
        view_415: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_115, [64, 128, 1024]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_140: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_137, view_415);  add_137 = view_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_526: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_140, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_526, [2], correction = 0, keepdim = True)
        getitem_121: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_38[0]
        getitem_122: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_42: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_42 = None
        scalar_tensor_43: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_43 = None
        full_default_42: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_54: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_526, getitem_122);  convert_element_type_526 = getitem_122 = None
        add_141: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_38: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_112: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_38);  sub_54 = rsqrt_38 = None
        mul_113: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, arg313_1);  mul_112 = arg313_1 = None
        add_142: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, arg314_1);  mul_113 = arg314_1 = None
        convert_element_type_527: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_416: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_527, [8192, 1024]);  convert_element_type_527 = None
        permute_216: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_116: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg316_1, view_416, permute_216);  arg316_1 = view_416 = permute_216 = None
        view_417: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_116, [64, 128, 1024]);  addmm_116 = None
        view_418: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_417, [64, 128, -1, 64]);  view_417 = None
        permute_217: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_114: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_217, 0.3535533905932738);  permute_217 = None
        expand_67: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_114, [64, 16, 128, 64]);  mul_114 = None
        clone_119: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_425: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_119, [1024, 128, 64]);  clone_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_419: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_218: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_117: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg318_1, view_419, permute_218);  arg318_1 = view_419 = permute_218 = None
        view_420: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_117, [64, 128, 1024]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_423: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_420, [64, 128, -1, 64]);  view_420 = None
        permute_220: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_222: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_220, [0, 1, 3, 2]);  permute_220 = None
        mul_115: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_222, 0.3535533905932738);  permute_222 = None
        expand_68: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_115, [64, 16, 64, 128]);  mul_115 = None
        clone_120: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_426: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [1024, 64, 128]);  clone_120 = None
        bmm_32: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_425, view_426);  view_425 = view_426 = None
        view_427: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [64, 16, 128, 128]);  bmm_32 = None
        eq_16: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_427, -inf)
        logical_not_32: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_16);  eq_16 = None
        any_17: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_32, -1, True);  logical_not_32 = None
        logical_not_33: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        full_default_43: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_539: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_427, torch.float32);  view_427 = None
        amax_16: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_539, [-1], True)
        sub_55: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_539, amax_16);  convert_element_type_539 = amax_16 = None
        exp_16: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_17: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_540: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        where_38: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_33, full_default_43, convert_element_type_540);  logical_not_33 = full_default_43 = convert_element_type_540 = None
        expand_69: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_38, [64, 16, 128, 128]);  where_38 = None
        view_428: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_69, [1024, 128, 128]);  expand_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_421: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_219: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_118: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg320_1, view_421, permute_219);  arg320_1 = view_421 = permute_219 = None
        view_422: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_118, [64, 128, 1024]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_424: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_422, [64, 128, -1, 64]);  view_422 = None
        permute_221: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_70: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_221, [64, 16, 128, 64]);  permute_221 = None
        clone_121: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_429: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [1024, 128, 64]);  clone_121 = None
        bmm_33: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_428, view_429);  view_428 = view_429 = None
        view_430: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [64, 16, 128, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_223: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_122: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_431: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [64, 128, -1]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_432: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_431, [8192, 1024]);  view_431 = None
        permute_224: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_119: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg322_1, view_432, permute_224);  arg322_1 = view_432 = permute_224 = None
        view_433: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_119, [64, 128, 1024]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_144: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_140, view_433);  add_140 = view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_546: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_546, [2], correction = 0, keepdim = True)
        getitem_123: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_39[0]
        getitem_124: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        sub_56: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_546, getitem_124);  convert_element_type_546 = getitem_124 = None
        add_145: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_123, 1e-05);  getitem_123 = None
        rsqrt_39: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        mul_116: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_39);  sub_56 = rsqrt_39 = None
        mul_117: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, arg323_1);  mul_116 = arg323_1 = None
        add_146: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_117, arg324_1);  mul_117 = arg324_1 = None
        convert_element_type_547: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_146, torch.bfloat16);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_434: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_547, [8192, 1024]);  convert_element_type_547 = None
        permute_225: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_120: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg326_1, view_434, permute_225);  arg326_1 = view_434 = permute_225 = None
        view_435: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_120, [64, 128, 4096]);  addmm_120 = None
        relu_16: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_435);  view_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_436: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_16, [8192, 4096]);  relu_16 = None
        permute_226: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_121: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg328_1, view_436, permute_226);  arg328_1 = view_436 = permute_226 = None
        view_437: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_121, [64, 128, 1024]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_147: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_144, view_437);  add_144 = view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_554: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_554, [2], correction = 0, keepdim = True)
        getitem_125: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_40[0]
        getitem_126: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        sub_57: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_554, getitem_126);  convert_element_type_554 = getitem_126 = None
        add_148: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_125, 1e-05);  getitem_125 = None
        rsqrt_40: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_118: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_40);  sub_57 = rsqrt_40 = None
        mul_119: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, arg329_1);  mul_118 = arg329_1 = None
        add_149: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_119, arg330_1);  mul_119 = arg330_1 = None
        convert_element_type_555: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_438: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_555, [8192, 1024])
        permute_227: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_122: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg332_1, view_438, permute_227);  arg332_1 = view_438 = permute_227 = None
        view_439: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_122, [64, 128, 1024]);  addmm_122 = None
        view_440: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_439, [64, 128, -1, 64]);  view_439 = None
        permute_228: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_441: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_555, [8192, 1024])
        permute_229: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_123: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg334_1, view_441, permute_229);  arg334_1 = view_441 = permute_229 = None
        view_442: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_123, [64, 128, 1024]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_445: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_442, [64, 128, -1, 64]);  view_442 = None
        permute_231: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_443: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_555, [8192, 1024]);  convert_element_type_555 = None
        permute_230: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_124: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg336_1, view_443, permute_230);  arg336_1 = view_443 = permute_230 = None
        view_444: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_124, [64, 128, 1024]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_446: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_444, [64, 128, -1, 64]);  view_444 = None
        permute_232: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_45: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_39: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_45, full_default_44);  full_default_45 = full_default_44 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_228, permute_231, permute_232, where_39, False, scale = 0.125);  permute_228 = permute_231 = permute_232 = where_39 = None
        getitem_127: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_233: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_127, [0, 2, 1, 3]);  getitem_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_447: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_233, [64, 128, -1]);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_448: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_447, [8192, 1024]);  view_447 = None
        permute_234: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_125: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg338_1, view_448, permute_234);  arg338_1 = view_448 = permute_234 = None
        view_449: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_125, [64, 128, 1024]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_150: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_147, view_449);  add_147 = view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_568: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_150, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_568, [2], correction = 0, keepdim = True)
        getitem_136: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_41[0]
        getitem_137: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_46: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_46 = None
        scalar_tensor_47: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_47 = None
        full_default_46: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_58: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_568, getitem_137);  convert_element_type_568 = getitem_137 = None
        add_151: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
        rsqrt_41: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_120: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_41);  sub_58 = rsqrt_41 = None
        mul_121: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, arg339_1);  mul_120 = arg339_1 = None
        add_152: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, arg340_1);  mul_121 = arg340_1 = None
        convert_element_type_569: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_450: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_569, [8192, 1024]);  convert_element_type_569 = None
        permute_235: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_126: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg342_1, view_450, permute_235);  arg342_1 = view_450 = permute_235 = None
        view_451: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_126, [64, 128, 1024]);  addmm_126 = None
        view_452: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_451, [64, 128, -1, 64]);  view_451 = None
        permute_236: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_122: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_236, 0.3535533905932738);  permute_236 = None
        expand_71: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_122, [64, 16, 128, 64]);  mul_122 = None
        clone_127: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_459: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [1024, 128, 64]);  clone_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_453: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_237: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_127: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg344_1, view_453, permute_237);  arg344_1 = view_453 = permute_237 = None
        view_454: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_127, [64, 128, 1024]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_457: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_454, [64, 128, -1, 64]);  view_454 = None
        permute_239: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_457, [0, 2, 1, 3]);  view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_241: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_239, [0, 1, 3, 2]);  permute_239 = None
        mul_123: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_241, 0.3535533905932738);  permute_241 = None
        expand_72: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_123, [64, 16, 64, 128]);  mul_123 = None
        clone_128: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_460: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [1024, 64, 128]);  clone_128 = None
        bmm_34: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_459, view_460);  view_459 = view_460 = None
        view_461: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [64, 16, 128, 128]);  bmm_34 = None
        eq_17: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_461, -inf)
        logical_not_34: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_17);  eq_17 = None
        any_18: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_34, -1, True);  logical_not_34 = None
        logical_not_35: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        full_default_47: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_581: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_461, torch.float32);  view_461 = None
        amax_17: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_581, [-1], True)
        sub_59: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_581, amax_17);  convert_element_type_581 = amax_17 = None
        exp_17: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
        sum_18: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_582: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        where_41: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_35, full_default_47, convert_element_type_582);  logical_not_35 = full_default_47 = convert_element_type_582 = None
        expand_73: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_41, [64, 16, 128, 128]);  where_41 = None
        view_462: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_73, [1024, 128, 128]);  expand_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_455: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_238: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_128: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg346_1, view_455, permute_238);  arg346_1 = view_455 = permute_238 = None
        view_456: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_128, [64, 128, 1024]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_458: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_456, [64, 128, -1, 64]);  view_456 = None
        permute_240: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_74: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_240, [64, 16, 128, 64]);  permute_240 = None
        clone_129: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_463: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [1024, 128, 64]);  clone_129 = None
        bmm_35: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_462, view_463);  view_462 = view_463 = None
        view_464: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [64, 16, 128, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_242: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
        clone_130: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_465: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [64, 128, -1]);  clone_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_466: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_465, [8192, 1024]);  view_465 = None
        permute_243: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_129: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg348_1, view_466, permute_243);  arg348_1 = view_466 = permute_243 = None
        view_467: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_129, [64, 128, 1024]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_154: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_150, view_467);  add_150 = view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_588: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_588, [2], correction = 0, keepdim = True)
        getitem_138: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_42[0]
        getitem_139: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        sub_60: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_588, getitem_139);  convert_element_type_588 = getitem_139 = None
        add_155: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
        rsqrt_42: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_124: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_42);  sub_60 = rsqrt_42 = None
        mul_125: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, arg349_1);  mul_124 = arg349_1 = None
        add_156: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, arg350_1);  mul_125 = arg350_1 = None
        convert_element_type_589: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_156, torch.bfloat16);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_468: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_589, [8192, 1024]);  convert_element_type_589 = None
        permute_244: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_130: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg352_1, view_468, permute_244);  arg352_1 = view_468 = permute_244 = None
        view_469: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_130, [64, 128, 4096]);  addmm_130 = None
        relu_17: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_469);  view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_470: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_17, [8192, 4096]);  relu_17 = None
        permute_245: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_131: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg354_1, view_470, permute_245);  arg354_1 = view_470 = permute_245 = None
        view_471: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_131, [64, 128, 1024]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_157: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_154, view_471);  add_154 = view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_596: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_596, [2], correction = 0, keepdim = True)
        getitem_140: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_43[0]
        getitem_141: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        sub_61: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_596, getitem_141);  convert_element_type_596 = getitem_141 = None
        add_158: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
        rsqrt_43: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        mul_126: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_43);  sub_61 = rsqrt_43 = None
        mul_127: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, arg355_1);  mul_126 = arg355_1 = None
        add_159: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, arg356_1);  mul_127 = arg356_1 = None
        convert_element_type_597: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_472: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_597, [8192, 1024])
        permute_246: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_132: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg358_1, view_472, permute_246);  arg358_1 = view_472 = permute_246 = None
        view_473: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_132, [64, 128, 1024]);  addmm_132 = None
        view_474: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_473, [64, 128, -1, 64]);  view_473 = None
        permute_247: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_475: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_597, [8192, 1024])
        permute_248: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_133: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg360_1, view_475, permute_248);  arg360_1 = view_475 = permute_248 = None
        view_476: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_133, [64, 128, 1024]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_479: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_476, [64, 128, -1, 64]);  view_476 = None
        permute_250: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_477: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_597, [8192, 1024]);  convert_element_type_597 = None
        permute_249: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_134: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg362_1, view_477, permute_249);  arg362_1 = view_477 = permute_249 = None
        view_478: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_134, [64, 128, 1024]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_480: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_478, [64, 128, -1, 64]);  view_478 = None
        permute_251: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_49: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_42: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_49, full_default_48);  full_default_49 = full_default_48 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_247, permute_250, permute_251, where_42, False, scale = 0.125);  permute_247 = permute_250 = permute_251 = where_42 = None
        getitem_142: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_252: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_481: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_252, [64, 128, -1]);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_482: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_481, [8192, 1024]);  view_481 = None
        permute_253: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_135: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg364_1, view_482, permute_253);  arg364_1 = view_482 = permute_253 = None
        view_483: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_135, [64, 128, 1024]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_160: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_157, view_483);  add_157 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_610: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_160, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_610, [2], correction = 0, keepdim = True)
        getitem_151: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_44[0]
        getitem_152: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_50: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_50 = None
        scalar_tensor_51: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_51 = None
        full_default_50: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_62: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_610, getitem_152);  convert_element_type_610 = getitem_152 = None
        add_161: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_151, 1e-05);  getitem_151 = None
        rsqrt_44: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        mul_128: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_44);  sub_62 = rsqrt_44 = None
        mul_129: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, arg365_1);  mul_128 = arg365_1 = None
        add_162: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_129, arg366_1);  mul_129 = arg366_1 = None
        convert_element_type_611: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_162, torch.bfloat16);  add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_484: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_611, [8192, 1024]);  convert_element_type_611 = None
        permute_254: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_136: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg368_1, view_484, permute_254);  arg368_1 = view_484 = permute_254 = None
        view_485: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_136, [64, 128, 1024]);  addmm_136 = None
        view_486: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [64, 128, -1, 64]);  view_485 = None
        permute_255: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_130: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_255, 0.3535533905932738);  permute_255 = None
        expand_75: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_130, [64, 16, 128, 64]);  mul_130 = None
        clone_135: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_493: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [1024, 128, 64]);  clone_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_487: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_256: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_137: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg370_1, view_487, permute_256);  arg370_1 = view_487 = permute_256 = None
        view_488: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_137, [64, 128, 1024]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_491: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_488, [64, 128, -1, 64]);  view_488 = None
        permute_258: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_260: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_258, [0, 1, 3, 2]);  permute_258 = None
        mul_131: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_260, 0.3535533905932738);  permute_260 = None
        expand_76: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_131, [64, 16, 64, 128]);  mul_131 = None
        clone_136: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_494: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [1024, 64, 128]);  clone_136 = None
        bmm_36: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_493, view_494);  view_493 = view_494 = None
        view_495: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [64, 16, 128, 128]);  bmm_36 = None
        eq_18: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_495, -inf)
        logical_not_36: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_18);  eq_18 = None
        any_19: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_36, -1, True);  logical_not_36 = None
        logical_not_37: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        full_default_51: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_623: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_495, torch.float32);  view_495 = None
        amax_18: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_623, [-1], True)
        sub_63: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_623, amax_18);  convert_element_type_623 = amax_18 = None
        exp_18: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
        sum_19: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_624: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        where_44: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_37, full_default_51, convert_element_type_624);  logical_not_37 = full_default_51 = convert_element_type_624 = None
        expand_77: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_44, [64, 16, 128, 128]);  where_44 = None
        view_496: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_77, [1024, 128, 128]);  expand_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_489: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_257: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
        addmm_138: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg372_1, view_489, permute_257);  arg372_1 = view_489 = permute_257 = None
        view_490: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_138, [64, 128, 1024]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_492: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_490, [64, 128, -1, 64]);  view_490 = None
        permute_259: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_78: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_259, [64, 16, 128, 64]);  permute_259 = None
        clone_137: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_497: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [1024, 128, 64]);  clone_137 = None
        bmm_37: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_496, view_497);  view_496 = view_497 = None
        view_498: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [64, 16, 128, 64]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_261: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_138: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_499: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [64, 128, -1]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_500: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_499, [8192, 1024]);  view_499 = None
        permute_262: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_139: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg374_1, view_500, permute_262);  arg374_1 = view_500 = permute_262 = None
        view_501: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_139, [64, 128, 1024]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_164: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_160, view_501);  add_160 = view_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_630: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_630, [2], correction = 0, keepdim = True)
        getitem_153: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_45[0]
        getitem_154: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        sub_64: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_630, getitem_154);  convert_element_type_630 = getitem_154 = None
        add_165: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_153, 1e-05);  getitem_153 = None
        rsqrt_45: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        mul_132: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_45);  sub_64 = rsqrt_45 = None
        mul_133: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, arg375_1);  mul_132 = arg375_1 = None
        add_166: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_133, arg376_1);  mul_133 = arg376_1 = None
        convert_element_type_631: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_166, torch.bfloat16);  add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_502: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_631, [8192, 1024]);  convert_element_type_631 = None
        permute_263: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_140: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg378_1, view_502, permute_263);  arg378_1 = view_502 = permute_263 = None
        view_503: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_140, [64, 128, 4096]);  addmm_140 = None
        relu_18: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_503);  view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_504: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_18, [8192, 4096]);  relu_18 = None
        permute_264: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_141: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg380_1, view_504, permute_264);  arg380_1 = view_504 = permute_264 = None
        view_505: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_141, [64, 128, 1024]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_167: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_164, view_505);  add_164 = view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_638: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_638, [2], correction = 0, keepdim = True)
        getitem_155: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_46[0]
        getitem_156: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        sub_65: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_638, getitem_156);  convert_element_type_638 = getitem_156 = None
        add_168: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_155, 1e-05);  getitem_155 = None
        rsqrt_46: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_134: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_46);  sub_65 = rsqrt_46 = None
        mul_135: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, arg381_1);  mul_134 = arg381_1 = None
        add_169: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_135, arg382_1);  mul_135 = arg382_1 = None
        convert_element_type_639: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_506: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_639, [8192, 1024])
        permute_265: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_142: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg384_1, view_506, permute_265);  arg384_1 = view_506 = permute_265 = None
        view_507: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_142, [64, 128, 1024]);  addmm_142 = None
        view_508: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_507, [64, 128, -1, 64]);  view_507 = None
        permute_266: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_509: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_639, [8192, 1024])
        permute_267: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_143: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg386_1, view_509, permute_267);  arg386_1 = view_509 = permute_267 = None
        view_510: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_143, [64, 128, 1024]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_513: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_510, [64, 128, -1, 64]);  view_510 = None
        permute_269: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_511: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_639, [8192, 1024]);  convert_element_type_639 = None
        permute_268: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        addmm_144: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg388_1, view_511, permute_268);  arg388_1 = view_511 = permute_268 = None
        view_512: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_144, [64, 128, 1024]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_514: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_512, [64, 128, -1, 64]);  view_512 = None
        permute_270: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_53: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_45: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_53, full_default_52);  full_default_53 = full_default_52 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_266, permute_269, permute_270, where_45, False, scale = 0.125);  permute_266 = permute_269 = permute_270 = where_45 = None
        getitem_157: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_271: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_157, [0, 2, 1, 3]);  getitem_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_515: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_271, [64, 128, -1]);  permute_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_516: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_515, [8192, 1024]);  view_515 = None
        permute_272: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        addmm_145: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg390_1, view_516, permute_272);  arg390_1 = view_516 = permute_272 = None
        view_517: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_145, [64, 128, 1024]);  addmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_170: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_167, view_517);  add_167 = view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_652: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_170, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_652, [2], correction = 0, keepdim = True)
        getitem_166: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_47[0]
        getitem_167: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_54: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_54 = None
        scalar_tensor_55: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_55 = None
        full_default_54: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_66: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_652, getitem_167);  convert_element_type_652 = getitem_167 = None
        add_171: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
        rsqrt_47: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        mul_136: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_47);  sub_66 = rsqrt_47 = None
        mul_137: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, arg391_1);  mul_136 = arg391_1 = None
        add_172: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, arg392_1);  mul_137 = arg392_1 = None
        convert_element_type_653: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_172, torch.bfloat16);  add_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_518: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_653, [8192, 1024]);  convert_element_type_653 = None
        permute_273: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
        addmm_146: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg394_1, view_518, permute_273);  arg394_1 = view_518 = permute_273 = None
        view_519: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_146, [64, 128, 1024]);  addmm_146 = None
        view_520: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_519, [64, 128, -1, 64]);  view_519 = None
        permute_274: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_138: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_274, 0.3535533905932738);  permute_274 = None
        expand_79: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_138, [64, 16, 128, 64]);  mul_138 = None
        clone_143: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_527: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_143, [1024, 128, 64]);  clone_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_521: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_275: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
        addmm_147: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg396_1, view_521, permute_275);  arg396_1 = view_521 = permute_275 = None
        view_522: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_147, [64, 128, 1024]);  addmm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_525: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_522, [64, 128, -1, 64]);  view_522 = None
        permute_277: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_525, [0, 2, 1, 3]);  view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_279: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_277, [0, 1, 3, 2]);  permute_277 = None
        mul_139: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_279, 0.3535533905932738);  permute_279 = None
        expand_80: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_139, [64, 16, 64, 128]);  mul_139 = None
        clone_144: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_528: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_144, [1024, 64, 128]);  clone_144 = None
        bmm_38: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_527, view_528);  view_527 = view_528 = None
        view_529: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [64, 16, 128, 128]);  bmm_38 = None
        eq_19: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_529, -inf)
        logical_not_38: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_19);  eq_19 = None
        any_20: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_38, -1, True);  logical_not_38 = None
        logical_not_39: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        full_default_55: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_665: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        amax_19: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_665, [-1], True)
        sub_67: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_665, amax_19);  convert_element_type_665 = amax_19 = None
        exp_19: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_20: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_666: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        where_47: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_39, full_default_55, convert_element_type_666);  logical_not_39 = full_default_55 = convert_element_type_666 = None
        expand_81: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_47, [64, 16, 128, 128]);  where_47 = None
        view_530: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_81, [1024, 128, 128]);  expand_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_523: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_276: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg397_1, [1, 0]);  arg397_1 = None
        addmm_148: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg398_1, view_523, permute_276);  arg398_1 = view_523 = permute_276 = None
        view_524: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_148, [64, 128, 1024]);  addmm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_526: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_524, [64, 128, -1, 64]);  view_524 = None
        permute_278: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_82: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_278, [64, 16, 128, 64]);  permute_278 = None
        clone_145: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_531: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [1024, 128, 64]);  clone_145 = None
        bmm_39: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_530, view_531);  view_530 = view_531 = None
        view_532: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [64, 16, 128, 64]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_280: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        clone_146: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_280, memory_format = torch.contiguous_format);  permute_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_533: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [64, 128, -1]);  clone_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_534: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_533, [8192, 1024]);  view_533 = None
        permute_281: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
        addmm_149: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg400_1, view_534, permute_281);  arg400_1 = view_534 = permute_281 = None
        view_535: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_149, [64, 128, 1024]);  addmm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_174: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_170, view_535);  add_170 = view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_672: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_672, [2], correction = 0, keepdim = True)
        getitem_168: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_48[0]
        getitem_169: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        sub_68: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_672, getitem_169);  convert_element_type_672 = getitem_169 = None
        add_175: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_168, 1e-05);  getitem_168 = None
        rsqrt_48: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        mul_140: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_48);  sub_68 = rsqrt_48 = None
        mul_141: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, arg401_1);  mul_140 = arg401_1 = None
        add_176: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_141, arg402_1);  mul_141 = arg402_1 = None
        convert_element_type_673: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16);  add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_536: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_673, [8192, 1024]);  convert_element_type_673 = None
        permute_282: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
        addmm_150: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg404_1, view_536, permute_282);  arg404_1 = view_536 = permute_282 = None
        view_537: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_150, [64, 128, 4096]);  addmm_150 = None
        relu_19: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_537);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_538: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_19, [8192, 4096]);  relu_19 = None
        permute_283: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        addmm_151: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg406_1, view_538, permute_283);  arg406_1 = view_538 = permute_283 = None
        view_539: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_151, [64, 128, 1024]);  addmm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_177: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_174, view_539);  add_174 = view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_680: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_680, [2], correction = 0, keepdim = True)
        getitem_170: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_49[0]
        getitem_171: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        sub_69: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_680, getitem_171);  convert_element_type_680 = getitem_171 = None
        add_178: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_170, 1e-05);  getitem_170 = None
        rsqrt_49: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_142: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_49);  sub_69 = rsqrt_49 = None
        mul_143: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_142, arg407_1);  mul_142 = arg407_1 = None
        add_179: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_143, arg408_1);  mul_143 = arg408_1 = None
        convert_element_type_681: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_540: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_681, [8192, 1024])
        permute_284: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        addmm_152: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg410_1, view_540, permute_284);  arg410_1 = view_540 = permute_284 = None
        view_541: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_152, [64, 128, 1024]);  addmm_152 = None
        view_542: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_541, [64, 128, -1, 64]);  view_541 = None
        permute_285: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_543: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_681, [8192, 1024])
        permute_286: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
        addmm_153: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg412_1, view_543, permute_286);  arg412_1 = view_543 = permute_286 = None
        view_544: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_153, [64, 128, 1024]);  addmm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_547: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_544, [64, 128, -1, 64]);  view_544 = None
        permute_288: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_547, [0, 2, 1, 3]);  view_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_545: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_681, [8192, 1024]);  convert_element_type_681 = None
        permute_287: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
        addmm_154: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg414_1, view_545, permute_287);  arg414_1 = view_545 = permute_287 = None
        view_546: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_154, [64, 128, 1024]);  addmm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_548: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_546, [64, 128, -1, 64]);  view_546 = None
        permute_289: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_57: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_57, full_default_56);  full_default_57 = full_default_56 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_285, permute_288, permute_289, where_48, False, scale = 0.125);  permute_285 = permute_288 = permute_289 = where_48 = None
        getitem_172: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_290: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_549: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_290, [64, 128, -1]);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_550: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_549, [8192, 1024]);  view_549 = None
        permute_291: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
        addmm_155: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg416_1, view_550, permute_291);  arg416_1 = view_550 = permute_291 = None
        view_551: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_155, [64, 128, 1024]);  addmm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_180: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_177, view_551);  add_177 = view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_694: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_180, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_694, [2], correction = 0, keepdim = True)
        getitem_181: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_50[0]
        getitem_182: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_58: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_58 = None
        scalar_tensor_59: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_59 = None
        full_default_58: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_70: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_694, getitem_182);  convert_element_type_694 = getitem_182 = None
        add_181: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_181, 1e-05);  getitem_181 = None
        rsqrt_50: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        mul_144: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_50);  sub_70 = rsqrt_50 = None
        mul_145: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, arg417_1);  mul_144 = arg417_1 = None
        add_182: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_145, arg418_1);  mul_145 = arg418_1 = None
        convert_element_type_695: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_182, torch.bfloat16);  add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_552: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_695, [8192, 1024]);  convert_element_type_695 = None
        permute_292: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
        addmm_156: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg420_1, view_552, permute_292);  arg420_1 = view_552 = permute_292 = None
        view_553: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_156, [64, 128, 1024]);  addmm_156 = None
        view_554: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_553, [64, 128, -1, 64]);  view_553 = None
        permute_293: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_146: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_293, 0.3535533905932738);  permute_293 = None
        expand_83: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_146, [64, 16, 128, 64]);  mul_146 = None
        clone_151: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_561: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [1024, 128, 64]);  clone_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_555: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_294: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
        addmm_157: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg422_1, view_555, permute_294);  arg422_1 = view_555 = permute_294 = None
        view_556: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_157, [64, 128, 1024]);  addmm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_559: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_556, [64, 128, -1, 64]);  view_556 = None
        permute_296: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_298: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_296, [0, 1, 3, 2]);  permute_296 = None
        mul_147: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_298, 0.3535533905932738);  permute_298 = None
        expand_84: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_147, [64, 16, 64, 128]);  mul_147 = None
        clone_152: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_562: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_152, [1024, 64, 128]);  clone_152 = None
        bmm_40: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_561, view_562);  view_561 = view_562 = None
        view_563: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [64, 16, 128, 128]);  bmm_40 = None
        eq_20: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_563, -inf)
        logical_not_40: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_20);  eq_20 = None
        any_21: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_40, -1, True);  logical_not_40 = None
        logical_not_41: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        full_default_59: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_707: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_563, torch.float32);  view_563 = None
        amax_20: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_707, [-1], True)
        sub_71: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_707, amax_20);  convert_element_type_707 = amax_20 = None
        exp_20: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        sum_21: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_708: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        where_50: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_41, full_default_59, convert_element_type_708);  logical_not_41 = full_default_59 = convert_element_type_708 = None
        expand_85: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_50, [64, 16, 128, 128]);  where_50 = None
        view_564: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_85, [1024, 128, 128]);  expand_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_557: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_295: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        addmm_158: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg424_1, view_557, permute_295);  arg424_1 = view_557 = permute_295 = None
        view_558: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_158, [64, 128, 1024]);  addmm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_560: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_558, [64, 128, -1, 64]);  view_558 = None
        permute_297: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_86: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_297, [64, 16, 128, 64]);  permute_297 = None
        clone_153: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_565: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [1024, 128, 64]);  clone_153 = None
        bmm_41: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_564, view_565);  view_564 = view_565 = None
        view_566: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [64, 16, 128, 64]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_299: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
        clone_154: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_567: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [64, 128, -1]);  clone_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_568: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_567, [8192, 1024]);  view_567 = None
        permute_300: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
        addmm_159: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg426_1, view_568, permute_300);  arg426_1 = view_568 = permute_300 = None
        view_569: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_159, [64, 128, 1024]);  addmm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_184: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_180, view_569);  add_180 = view_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_714: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_714, [2], correction = 0, keepdim = True)
        getitem_183: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_51[0]
        getitem_184: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        sub_72: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_714, getitem_184);  convert_element_type_714 = getitem_184 = None
        add_185: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_183, 1e-05);  getitem_183 = None
        rsqrt_51: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        mul_148: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_51);  sub_72 = rsqrt_51 = None
        mul_149: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, arg427_1);  mul_148 = arg427_1 = None
        add_186: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, arg428_1);  mul_149 = arg428_1 = None
        convert_element_type_715: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16);  add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_570: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_715, [8192, 1024]);  convert_element_type_715 = None
        permute_301: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
        addmm_160: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg430_1, view_570, permute_301);  arg430_1 = view_570 = permute_301 = None
        view_571: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_160, [64, 128, 4096]);  addmm_160 = None
        relu_20: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_571);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_572: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_20, [8192, 4096]);  relu_20 = None
        permute_302: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        addmm_161: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg432_1, view_572, permute_302);  arg432_1 = view_572 = permute_302 = None
        view_573: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_161, [64, 128, 1024]);  addmm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_187: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_184, view_573);  add_184 = view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_722: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_187, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_722, [2], correction = 0, keepdim = True)
        getitem_185: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_52[0]
        getitem_186: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        sub_73: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_722, getitem_186);  convert_element_type_722 = getitem_186 = None
        add_188: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_185, 1e-05);  getitem_185 = None
        rsqrt_52: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_150: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_52);  sub_73 = rsqrt_52 = None
        mul_151: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, arg433_1);  mul_150 = arg433_1 = None
        add_189: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, arg434_1);  mul_151 = arg434_1 = None
        convert_element_type_723: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_574: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_723, [8192, 1024])
        permute_303: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_162: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg436_1, view_574, permute_303);  arg436_1 = view_574 = permute_303 = None
        view_575: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_162, [64, 128, 1024]);  addmm_162 = None
        view_576: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_575, [64, 128, -1, 64]);  view_575 = None
        permute_304: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_577: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_723, [8192, 1024])
        permute_305: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg437_1, [1, 0]);  arg437_1 = None
        addmm_163: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg438_1, view_577, permute_305);  arg438_1 = view_577 = permute_305 = None
        view_578: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_163, [64, 128, 1024]);  addmm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_581: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_578, [64, 128, -1, 64]);  view_578 = None
        permute_307: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_581, [0, 2, 1, 3]);  view_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_579: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_723, [8192, 1024]);  convert_element_type_723 = None
        permute_306: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg439_1, [1, 0]);  arg439_1 = None
        addmm_164: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg440_1, view_579, permute_306);  arg440_1 = view_579 = permute_306 = None
        view_580: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_164, [64, 128, 1024]);  addmm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_582: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_580, [64, 128, -1, 64]);  view_580 = None
        permute_308: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_61: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_51: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_61, full_default_60);  full_default_61 = full_default_60 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_304, permute_307, permute_308, where_51, False, scale = 0.125);  permute_304 = permute_307 = permute_308 = where_51 = None
        getitem_187: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_309: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_187, [0, 2, 1, 3]);  getitem_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_583: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_309, [64, 128, -1]);  permute_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_584: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_583, [8192, 1024]);  view_583 = None
        permute_310: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_165: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg442_1, view_584, permute_310);  arg442_1 = view_584 = permute_310 = None
        view_585: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_165, [64, 128, 1024]);  addmm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_190: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_187, view_585);  add_187 = view_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_736: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_190, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_736, [2], correction = 0, keepdim = True)
        getitem_196: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_53[0]
        getitem_197: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_53[1];  var_mean_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_62: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_62 = None
        scalar_tensor_63: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_63 = None
        full_default_62: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_74: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_736, getitem_197);  convert_element_type_736 = getitem_197 = None
        add_191: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_53: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        mul_152: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_53);  sub_74 = rsqrt_53 = None
        mul_153: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, arg443_1);  mul_152 = arg443_1 = None
        add_192: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_153, arg444_1);  mul_153 = arg444_1 = None
        convert_element_type_737: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_192, torch.bfloat16);  add_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_586: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_737, [8192, 1024]);  convert_element_type_737 = None
        permute_311: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
        addmm_166: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg446_1, view_586, permute_311);  arg446_1 = view_586 = permute_311 = None
        view_587: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_166, [64, 128, 1024]);  addmm_166 = None
        view_588: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_587, [64, 128, -1, 64]);  view_587 = None
        permute_312: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_154: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_312, 0.3535533905932738);  permute_312 = None
        expand_87: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_154, [64, 16, 128, 64]);  mul_154 = None
        clone_159: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_595: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_159, [1024, 128, 64]);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_589: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_313: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        addmm_167: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg448_1, view_589, permute_313);  arg448_1 = view_589 = permute_313 = None
        view_590: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_167, [64, 128, 1024]);  addmm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_593: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_590, [64, 128, -1, 64]);  view_590 = None
        permute_315: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_317: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_315, [0, 1, 3, 2]);  permute_315 = None
        mul_155: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_317, 0.3535533905932738);  permute_317 = None
        expand_88: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_155, [64, 16, 64, 128]);  mul_155 = None
        clone_160: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_596: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_160, [1024, 64, 128]);  clone_160 = None
        bmm_42: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_595, view_596);  view_595 = view_596 = None
        view_597: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [64, 16, 128, 128]);  bmm_42 = None
        eq_21: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_597, -inf)
        logical_not_42: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_21);  eq_21 = None
        any_22: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_42, -1, True);  logical_not_42 = None
        logical_not_43: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        full_default_63: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_749: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_597, torch.float32);  view_597 = None
        amax_21: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_749, [-1], True)
        sub_75: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_749, amax_21);  convert_element_type_749 = amax_21 = None
        exp_21: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
        sum_22: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_750: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        where_53: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_43, full_default_63, convert_element_type_750);  logical_not_43 = full_default_63 = convert_element_type_750 = None
        expand_89: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_53, [64, 16, 128, 128]);  where_53 = None
        view_598: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_89, [1024, 128, 128]);  expand_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_591: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_314: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
        addmm_168: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg450_1, view_591, permute_314);  arg450_1 = view_591 = permute_314 = None
        view_592: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_168, [64, 128, 1024]);  addmm_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_594: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_592, [64, 128, -1, 64]);  view_592 = None
        permute_316: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_90: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_316, [64, 16, 128, 64]);  permute_316 = None
        clone_161: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_599: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [1024, 128, 64]);  clone_161 = None
        bmm_43: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_598, view_599);  view_598 = view_599 = None
        view_600: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [64, 16, 128, 64]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_318: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
        clone_162: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_601: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [64, 128, -1]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_602: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_601, [8192, 1024]);  view_601 = None
        permute_319: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg451_1, [1, 0]);  arg451_1 = None
        addmm_169: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg452_1, view_602, permute_319);  arg452_1 = view_602 = permute_319 = None
        view_603: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_169, [64, 128, 1024]);  addmm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_194: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_190, view_603);  add_190 = view_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_756: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_194, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_756, [2], correction = 0, keepdim = True)
        getitem_198: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_54[0]
        getitem_199: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_54[1];  var_mean_54 = None
        sub_76: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_756, getitem_199);  convert_element_type_756 = getitem_199 = None
        add_195: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_54: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        mul_156: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_54);  sub_76 = rsqrt_54 = None
        mul_157: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, arg453_1);  mul_156 = arg453_1 = None
        add_196: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_157, arg454_1);  mul_157 = arg454_1 = None
        convert_element_type_757: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_604: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_757, [8192, 1024]);  convert_element_type_757 = None
        permute_320: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg455_1, [1, 0]);  arg455_1 = None
        addmm_170: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg456_1, view_604, permute_320);  arg456_1 = view_604 = permute_320 = None
        view_605: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_170, [64, 128, 4096]);  addmm_170 = None
        relu_21: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_605);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_606: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_21, [8192, 4096]);  relu_21 = None
        permute_321: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
        addmm_171: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg458_1, view_606, permute_321);  arg458_1 = view_606 = permute_321 = None
        view_607: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_171, [64, 128, 1024]);  addmm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_197: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_194, view_607);  add_194 = view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_764: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_197, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_764, [2], correction = 0, keepdim = True)
        getitem_200: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_55[0]
        getitem_201: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_55[1];  var_mean_55 = None
        sub_77: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_764, getitem_201);  convert_element_type_764 = getitem_201 = None
        add_198: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_55: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        mul_158: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_55);  sub_77 = rsqrt_55 = None
        mul_159: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, arg459_1);  mul_158 = arg459_1 = None
        add_199: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_159, arg460_1);  mul_159 = arg460_1 = None
        convert_element_type_765: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16);  add_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_608: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_765, [8192, 1024])
        permute_322: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        addmm_172: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg462_1, view_608, permute_322);  arg462_1 = view_608 = permute_322 = None
        view_609: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_172, [64, 128, 1024]);  addmm_172 = None
        view_610: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_609, [64, 128, -1, 64]);  view_609 = None
        permute_323: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_611: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_765, [8192, 1024])
        permute_324: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
        addmm_173: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg464_1, view_611, permute_324);  arg464_1 = view_611 = permute_324 = None
        view_612: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_173, [64, 128, 1024]);  addmm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_615: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_612, [64, 128, -1, 64]);  view_612 = None
        permute_326: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_613: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_765, [8192, 1024]);  convert_element_type_765 = None
        permute_325: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
        addmm_174: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg466_1, view_613, permute_325);  arg466_1 = view_613 = permute_325 = None
        view_614: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_174, [64, 128, 1024]);  addmm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_616: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_614, [64, 128, -1, 64]);  view_614 = None
        permute_327: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_65: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_54: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_65, full_default_64);  full_default_65 = full_default_64 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_323, permute_326, permute_327, where_54, False, scale = 0.125);  permute_323 = permute_326 = permute_327 = where_54 = None
        getitem_202: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_328: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_617: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_328, [64, 128, -1]);  permute_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_618: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_617, [8192, 1024]);  view_617 = None
        permute_329: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg467_1, [1, 0]);  arg467_1 = None
        addmm_175: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg468_1, view_618, permute_329);  arg468_1 = view_618 = permute_329 = None
        view_619: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_175, [64, 128, 1024]);  addmm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_200: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_197, view_619);  add_197 = view_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_778: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_200, torch.float32)
        var_mean_56 = torch.ops.aten.var_mean.correction(convert_element_type_778, [2], correction = 0, keepdim = True)
        getitem_211: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_56[0]
        getitem_212: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_56[1];  var_mean_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_66: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_66 = None
        scalar_tensor_67: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_67 = None
        full_default_66: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_78: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_778, getitem_212);  convert_element_type_778 = getitem_212 = None
        add_201: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_211, 1e-05);  getitem_211 = None
        rsqrt_56: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        mul_160: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_56);  sub_78 = rsqrt_56 = None
        mul_161: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, arg469_1);  mul_160 = arg469_1 = None
        add_202: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, arg470_1);  mul_161 = arg470_1 = None
        convert_element_type_779: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16);  add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_620: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_779, [8192, 1024]);  convert_element_type_779 = None
        permute_330: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
        addmm_176: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg472_1, view_620, permute_330);  arg472_1 = view_620 = permute_330 = None
        view_621: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_176, [64, 128, 1024]);  addmm_176 = None
        view_622: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_621, [64, 128, -1, 64]);  view_621 = None
        permute_331: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_622, [0, 2, 1, 3]);  view_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_162: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_331, 0.3535533905932738);  permute_331 = None
        expand_91: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_162, [64, 16, 128, 64]);  mul_162 = None
        clone_167: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_629: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_167, [1024, 128, 64]);  clone_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_623: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_332: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg473_1, [1, 0]);  arg473_1 = None
        addmm_177: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg474_1, view_623, permute_332);  arg474_1 = view_623 = permute_332 = None
        view_624: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_177, [64, 128, 1024]);  addmm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_627: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_624, [64, 128, -1, 64]);  view_624 = None
        permute_334: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_336: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_334, [0, 1, 3, 2]);  permute_334 = None
        mul_163: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_336, 0.3535533905932738);  permute_336 = None
        expand_92: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_163, [64, 16, 64, 128]);  mul_163 = None
        clone_168: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_630: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_168, [1024, 64, 128]);  clone_168 = None
        bmm_44: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_629, view_630);  view_629 = view_630 = None
        view_631: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [64, 16, 128, 128]);  bmm_44 = None
        eq_22: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_631, -inf)
        logical_not_44: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_22);  eq_22 = None
        any_23: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_44, -1, True);  logical_not_44 = None
        logical_not_45: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        full_default_67: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_791: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_631, torch.float32);  view_631 = None
        amax_22: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_791, [-1], True)
        sub_79: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_791, amax_22);  convert_element_type_791 = amax_22 = None
        exp_22: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
        sum_23: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_792: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        where_56: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_45, full_default_67, convert_element_type_792);  logical_not_45 = full_default_67 = convert_element_type_792 = None
        expand_93: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_56, [64, 16, 128, 128]);  where_56 = None
        view_632: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_93, [1024, 128, 128]);  expand_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_625: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_333: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
        addmm_178: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg476_1, view_625, permute_333);  arg476_1 = view_625 = permute_333 = None
        view_626: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_178, [64, 128, 1024]);  addmm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_628: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_626, [64, 128, -1, 64]);  view_626 = None
        permute_335: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_628, [0, 2, 1, 3]);  view_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_94: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_335, [64, 16, 128, 64]);  permute_335 = None
        clone_169: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_633: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [1024, 128, 64]);  clone_169 = None
        bmm_45: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_632, view_633);  view_632 = view_633 = None
        view_634: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [64, 16, 128, 64]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_337: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None
        clone_170: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_635: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [64, 128, -1]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_636: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_635, [8192, 1024]);  view_635 = None
        permute_338: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_179: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg478_1, view_636, permute_338);  arg478_1 = view_636 = permute_338 = None
        view_637: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_179, [64, 128, 1024]);  addmm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_204: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_200, view_637);  add_200 = view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_798: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.float32)
        var_mean_57 = torch.ops.aten.var_mean.correction(convert_element_type_798, [2], correction = 0, keepdim = True)
        getitem_213: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_57[0]
        getitem_214: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_57[1];  var_mean_57 = None
        sub_80: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_798, getitem_214);  convert_element_type_798 = getitem_214 = None
        add_205: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_213, 1e-05);  getitem_213 = None
        rsqrt_57: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        mul_164: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_57);  sub_80 = rsqrt_57 = None
        mul_165: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, arg479_1);  mul_164 = arg479_1 = None
        add_206: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, arg480_1);  mul_165 = arg480_1 = None
        convert_element_type_799: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_206, torch.bfloat16);  add_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_638: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_799, [8192, 1024]);  convert_element_type_799 = None
        permute_339: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
        addmm_180: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg482_1, view_638, permute_339);  arg482_1 = view_638 = permute_339 = None
        view_639: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_180, [64, 128, 4096]);  addmm_180 = None
        relu_22: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_639);  view_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_640: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_22, [8192, 4096]);  relu_22 = None
        permute_340: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
        addmm_181: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg484_1, view_640, permute_340);  arg484_1 = view_640 = permute_340 = None
        view_641: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_181, [64, 128, 1024]);  addmm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_207: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_204, view_641);  add_204 = view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_806: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_207, torch.float32)
        var_mean_58 = torch.ops.aten.var_mean.correction(convert_element_type_806, [2], correction = 0, keepdim = True)
        getitem_215: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_58[0]
        getitem_216: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_58[1];  var_mean_58 = None
        sub_81: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_806, getitem_216);  convert_element_type_806 = getitem_216 = None
        add_208: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_215, 1e-05);  getitem_215 = None
        rsqrt_58: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        mul_166: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_58);  sub_81 = rsqrt_58 = None
        mul_167: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, arg485_1);  mul_166 = arg485_1 = None
        add_209: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_167, arg486_1);  mul_167 = arg486_1 = None
        convert_element_type_807: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16);  add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_642: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_807, [8192, 1024])
        permute_341: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
        addmm_182: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg488_1, view_642, permute_341);  arg488_1 = view_642 = permute_341 = None
        view_643: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_182, [64, 128, 1024]);  addmm_182 = None
        view_644: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_643, [64, 128, -1, 64]);  view_643 = None
        permute_342: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_644, [0, 2, 1, 3]);  view_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_645: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_807, [8192, 1024])
        permute_343: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
        addmm_183: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg490_1, view_645, permute_343);  arg490_1 = view_645 = permute_343 = None
        view_646: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_183, [64, 128, 1024]);  addmm_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_649: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_646, [64, 128, -1, 64]);  view_646 = None
        permute_345: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_647: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_807, [8192, 1024]);  convert_element_type_807 = None
        permute_344: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg491_1, [1, 0]);  arg491_1 = None
        addmm_184: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg492_1, view_647, permute_344);  arg492_1 = view_647 = permute_344 = None
        view_648: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_184, [64, 128, 1024]);  addmm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_650: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_648, [64, 128, -1, 64]);  view_648 = None
        permute_346: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_650, [0, 2, 1, 3]);  view_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_69: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_57: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_49, full_default_69, full_default_68);  expand_49 = full_default_69 = full_default_68 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_342, permute_345, permute_346, where_57, False, scale = 0.125);  permute_342 = permute_345 = permute_346 = where_57 = None
        getitem_217: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_347: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_217, [0, 2, 1, 3]);  getitem_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_651: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_347, [64, 128, -1]);  permute_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_652: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_651, [8192, 1024]);  view_651 = None
        permute_348: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_185: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg494_1, view_652, permute_348);  arg494_1 = view_652 = permute_348 = None
        view_653: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_185, [64, 128, 1024]);  addmm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_210: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_207, view_653);  add_207 = view_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_820: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_210, torch.float32)
        var_mean_59 = torch.ops.aten.var_mean.correction(convert_element_type_820, [2], correction = 0, keepdim = True)
        getitem_226: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_59[0]
        getitem_227: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_59[1];  var_mean_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_70: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_70 = None
        scalar_tensor_71: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_71 = None
        full_default_70: "bf16[64, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_82: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_820, getitem_227);  convert_element_type_820 = getitem_227 = None
        add_211: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_59: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        mul_168: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_59);  sub_82 = rsqrt_59 = None
        mul_169: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, arg495_1);  mul_168 = arg495_1 = None
        add_212: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, arg496_1);  mul_169 = arg496_1 = None
        convert_element_type_821: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_212, torch.bfloat16);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_654: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_821, [8192, 1024]);  convert_element_type_821 = None
        permute_349: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
        addmm_186: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg498_1, view_654, permute_349);  arg498_1 = view_654 = permute_349 = None
        view_655: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_186, [64, 128, 1024]);  addmm_186 = None
        view_656: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_655, [64, 128, -1, 64]);  view_655 = None
        permute_350: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_656, [0, 2, 1, 3]);  view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_170: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_350, 0.3535533905932738);  permute_350 = None
        expand_95: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_170, [64, 16, 128, 64]);  mul_170 = None
        clone_175: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_663: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_175, [1024, 128, 64]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_657: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_351: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        addmm_187: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg500_1, view_657, permute_351);  arg500_1 = view_657 = permute_351 = None
        view_658: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_187, [64, 128, 1024]);  addmm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_661: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_658, [64, 128, -1, 64]);  view_658 = None
        permute_353: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_661, [0, 2, 1, 3]);  view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_355: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(permute_353, [0, 1, 3, 2]);  permute_353 = None
        mul_171: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.mul.Scalar(permute_355, 0.3535533905932738);  permute_355 = None
        expand_96: "bf16[64, 16, 64, 128][131072, 64, 1, 1024]cuda:0" = torch.ops.aten.expand.default(mul_171, [64, 16, 64, 128]);  mul_171 = None
        clone_176: "bf16[64, 16, 64, 128][131072, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_664: "bf16[1024, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_176, [1024, 64, 128]);  clone_176 = None
        bmm_46: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_663, view_664);  view_663 = view_664 = None
        view_665: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [64, 16, 128, 128]);  bmm_46 = None
        eq_23: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_665, -inf)
        logical_not_46: "b8[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_23);  eq_23 = None
        any_24: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_46, -1, True);  logical_not_46 = None
        logical_not_47: "b8[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        full_default_71: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_833: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_665, torch.float32);  view_665 = None
        amax_23: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_833, [-1], True)
        sub_83: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_833, amax_23);  convert_element_type_833 = amax_23 = None
        exp_23: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_83);  sub_83 = None
        sum_24: "f32[64, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_834: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        where_59: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_47, full_default_71, convert_element_type_834);  logical_not_47 = full_default_71 = convert_element_type_834 = None
        expand_97: "bf16[64, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_59, [64, 16, 128, 128]);  where_59 = None
        view_666: "bf16[1024, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_97, [1024, 128, 128]);  expand_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        view_659: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [8192, 1024])
        permute_352: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
        addmm_188: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg502_1, view_659, permute_352);  arg502_1 = view_659 = permute_352 = None
        view_660: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_188, [64, 128, 1024]);  addmm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_662: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_660, [64, 128, -1, 64]);  view_660 = None
        permute_354: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_662, [0, 2, 1, 3]);  view_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_98: "bf16[64, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.expand.default(permute_354, [64, 16, 128, 64]);  permute_354 = None
        clone_177: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_667: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [1024, 128, 64]);  clone_177 = None
        bmm_47: "bf16[1024, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_666, view_667);  view_666 = view_667 = None
        view_668: "bf16[64, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [64, 16, 128, 64]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_356: "bf16[64, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_668, [0, 2, 1, 3]);  view_668 = None
        clone_178: "bf16[64, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_669: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [64, 128, -1]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_670: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_669, [8192, 1024]);  view_669 = None
        permute_357: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg503_1, [1, 0]);  arg503_1 = None
        addmm_189: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg504_1, view_670, permute_357);  arg504_1 = view_670 = permute_357 = None
        view_671: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_189, [64, 128, 1024]);  addmm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_214: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_210, view_671);  add_210 = view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_840: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.float32)
        var_mean_60 = torch.ops.aten.var_mean.correction(convert_element_type_840, [2], correction = 0, keepdim = True)
        getitem_228: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_60[0]
        getitem_229: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_60[1];  var_mean_60 = None
        sub_84: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_840, getitem_229);  convert_element_type_840 = getitem_229 = None
        add_215: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_228, 1e-05);  getitem_228 = None
        rsqrt_60: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        mul_172: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_60);  sub_84 = rsqrt_60 = None
        mul_173: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, arg505_1);  mul_172 = arg505_1 = None
        add_216: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_173, arg506_1);  mul_173 = arg506_1 = None
        convert_element_type_841: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_216, torch.bfloat16);  add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_672: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_841, [8192, 1024]);  convert_element_type_841 = None
        permute_358: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
        addmm_190: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg508_1, view_672, permute_358);  arg508_1 = view_672 = permute_358 = None
        view_673: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_190, [64, 128, 4096]);  addmm_190 = None
        relu_23: "bf16[64, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.relu.default(view_673);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_674: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(relu_23, [8192, 4096]);  relu_23 = None
        permute_359: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        addmm_191: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg510_1, view_674, permute_359);  arg510_1 = view_674 = permute_359 = None
        view_675: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_191, [64, 128, 1024]);  addmm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_217: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_214, view_675);  add_214 = view_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:707 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_848: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_217, torch.float32);  add_217 = None
        var_mean_61 = torch.ops.aten.var_mean.correction(convert_element_type_848, [2], correction = 0, keepdim = True)
        getitem_230: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_61[0]
        getitem_231: "f32[64, 128, 1][128, 1, 1]cuda:0" = var_mean_61[1];  var_mean_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:908 in forward, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_679: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(arg0_1, [-1]);  arg0_1 = None
        ne_3: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_679, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:707 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_85: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_848, getitem_231);  convert_element_type_848 = getitem_231 = None
        add_218: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_230, 1e-05);  getitem_230 = None
        rsqrt_61: "f32[64, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        mul_174: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_61);  sub_85 = rsqrt_61 = None
        mul_175: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, arg511_1);  mul_174 = arg511_1 = None
        add_219: "f32[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_175, arg512_1);  mul_175 = arg512_1 = None
        convert_element_type_849: "bf16[64, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:901 in forward, code: lm_logits = self.lm_head(outputs.last_hidden_state)
        view_676: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_849, [8192, 1024]);  convert_element_type_849 = None
        permute_360: "bf16[1024, 128112][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm: "bf16[8192, 128112][128112, 1]cuda:0" = torch.ops.aten.mm.default(view_676, permute_360);  view_676 = permute_360 = None
        view_677: "bf16[64, 128, 128112][16398336, 128112, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [64, 128, 128112]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:908 in forward, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_678: "bf16[8192, 128112][128112, 1]cuda:0" = torch.ops.aten.reshape.default(view_677, [-1, 128112])
        convert_element_type_852: "f32[8192, 128112][128112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_678, torch.float32);  view_678 = None
        amax_24: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_852, [1], True)
        sub_86: "f32[8192, 128112][128112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_852, amax_24);  convert_element_type_852 = amax_24 = None
        exp_24: "f32[8192, 128112][128112, 1]cuda:0" = torch.ops.aten.exp.default(sub_86)
        sum_25: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_87: "f32[8192, 128112][128112, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, log);  sub_86 = log = None
        convert_element_type_853: "bf16[8192, 128112][128112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_87, torch.bfloat16);  sub_87 = None
        ne_2: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_679, -100)
        full_default_72: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_60: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne_2, view_679, full_default_72);  ne_2 = full_default_72 = None
        unsqueeze_12: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_60, 1);  where_60 = None
        gather: "bf16[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_853, 1, unsqueeze_12);  convert_element_type_853 = unsqueeze_12 = None
        squeeze: "bf16[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_73: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_61: "bf16[8192][1]cuda:0" = torch.ops.aten.where.self(ne_3, neg, full_default_73);  ne_3 = neg = full_default_73 = None
        sum_27: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_61);  where_61 = None
        ne_4: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_679, -100);  view_679 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_4);  ne_4 = None
        convert_element_type_854: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.bfloat16);  sum_26 = None
        div_24: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_854);  sum_27 = convert_element_type_854 = None
        return (div_24, view_677, convert_element_type_340)

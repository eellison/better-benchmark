class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 128][128, 1]cuda:0", arg1_1: "bf16[8008, 2560][2560, 1]cuda:0", arg2_1: "bf16[128, 2560][2560, 1]cuda:0", arg3_1: "bf16[2560][1]cuda:0", arg4_1: "bf16[2560][1]cuda:0", arg5_1: "bf16[2560, 2560][2560, 1]cuda:0", arg6_1: "bf16[2560][1]cuda:0", arg7_1: "bf16[2560, 2560][2560, 1]cuda:0", arg8_1: "bf16[2560][1]cuda:0", arg9_1: "bf16[2560, 2560][2560, 1]cuda:0", arg10_1: "bf16[2560][1]cuda:0", arg11_1: "bf16[2560, 2560][2560, 1]cuda:0", arg12_1: "bf16[2560][1]cuda:0", arg13_1: "bf16[2560][1]cuda:0", arg14_1: "bf16[2560][1]cuda:0", arg15_1: "bf16[10240, 2560][2560, 1]cuda:0", arg16_1: "bf16[10240][1]cuda:0", arg17_1: "bf16[2560, 10240][10240, 1]cuda:0", arg18_1: "bf16[2560][1]cuda:0", arg19_1: "bf16[2560][1]cuda:0", arg20_1: "bf16[2560][1]cuda:0", arg21_1: "bf16[2560, 2560][2560, 1]cuda:0", arg22_1: "bf16[2560][1]cuda:0", arg23_1: "bf16[2560, 2560][2560, 1]cuda:0", arg24_1: "bf16[2560][1]cuda:0", arg25_1: "bf16[2560, 2560][2560, 1]cuda:0", arg26_1: "bf16[2560][1]cuda:0", arg27_1: "bf16[2560, 2560][2560, 1]cuda:0", arg28_1: "bf16[2560][1]cuda:0", arg29_1: "bf16[2560][1]cuda:0", arg30_1: "bf16[2560][1]cuda:0", arg31_1: "bf16[10240, 2560][2560, 1]cuda:0", arg32_1: "bf16[10240][1]cuda:0", arg33_1: "bf16[2560, 10240][10240, 1]cuda:0", arg34_1: "bf16[2560][1]cuda:0", arg35_1: "bf16[2560][1]cuda:0", arg36_1: "bf16[2560][1]cuda:0", arg37_1: "bf16[2560, 2560][2560, 1]cuda:0", arg38_1: "bf16[2560][1]cuda:0", arg39_1: "bf16[2560, 2560][2560, 1]cuda:0", arg40_1: "bf16[2560][1]cuda:0", arg41_1: "bf16[2560, 2560][2560, 1]cuda:0", arg42_1: "bf16[2560][1]cuda:0", arg43_1: "bf16[2560, 2560][2560, 1]cuda:0", arg44_1: "bf16[2560][1]cuda:0", arg45_1: "bf16[2560][1]cuda:0", arg46_1: "bf16[2560][1]cuda:0", arg47_1: "bf16[10240, 2560][2560, 1]cuda:0", arg48_1: "bf16[10240][1]cuda:0", arg49_1: "bf16[2560, 10240][10240, 1]cuda:0", arg50_1: "bf16[2560][1]cuda:0", arg51_1: "bf16[2560][1]cuda:0", arg52_1: "bf16[2560][1]cuda:0", arg53_1: "bf16[2560, 2560][2560, 1]cuda:0", arg54_1: "bf16[2560][1]cuda:0", arg55_1: "bf16[2560, 2560][2560, 1]cuda:0", arg56_1: "bf16[2560][1]cuda:0", arg57_1: "bf16[2560, 2560][2560, 1]cuda:0", arg58_1: "bf16[2560][1]cuda:0", arg59_1: "bf16[2560, 2560][2560, 1]cuda:0", arg60_1: "bf16[2560][1]cuda:0", arg61_1: "bf16[2560][1]cuda:0", arg62_1: "bf16[2560][1]cuda:0", arg63_1: "bf16[10240, 2560][2560, 1]cuda:0", arg64_1: "bf16[10240][1]cuda:0", arg65_1: "bf16[2560, 10240][10240, 1]cuda:0", arg66_1: "bf16[2560][1]cuda:0", arg67_1: "bf16[2560][1]cuda:0", arg68_1: "bf16[2560][1]cuda:0", arg69_1: "bf16[2560, 2560][2560, 1]cuda:0", arg70_1: "bf16[2560][1]cuda:0", arg71_1: "bf16[2560, 2560][2560, 1]cuda:0", arg72_1: "bf16[2560][1]cuda:0", arg73_1: "bf16[2560, 2560][2560, 1]cuda:0", arg74_1: "bf16[2560][1]cuda:0", arg75_1: "bf16[2560, 2560][2560, 1]cuda:0", arg76_1: "bf16[2560][1]cuda:0", arg77_1: "bf16[2560][1]cuda:0", arg78_1: "bf16[2560][1]cuda:0", arg79_1: "bf16[10240, 2560][2560, 1]cuda:0", arg80_1: "bf16[10240][1]cuda:0", arg81_1: "bf16[2560, 10240][10240, 1]cuda:0", arg82_1: "bf16[2560][1]cuda:0", arg83_1: "bf16[2560][1]cuda:0", arg84_1: "bf16[2560][1]cuda:0", arg85_1: "bf16[2560, 2560][2560, 1]cuda:0", arg86_1: "bf16[2560][1]cuda:0", arg87_1: "bf16[2560, 2560][2560, 1]cuda:0", arg88_1: "bf16[2560][1]cuda:0", arg89_1: "bf16[2560, 2560][2560, 1]cuda:0", arg90_1: "bf16[2560][1]cuda:0", arg91_1: "bf16[2560, 2560][2560, 1]cuda:0", arg92_1: "bf16[2560][1]cuda:0", arg93_1: "bf16[2560][1]cuda:0", arg94_1: "bf16[2560][1]cuda:0", arg95_1: "bf16[10240, 2560][2560, 1]cuda:0", arg96_1: "bf16[10240][1]cuda:0", arg97_1: "bf16[2560, 10240][10240, 1]cuda:0", arg98_1: "bf16[2560][1]cuda:0", arg99_1: "bf16[2560][1]cuda:0", arg100_1: "bf16[2560][1]cuda:0", arg101_1: "bf16[2560, 2560][2560, 1]cuda:0", arg102_1: "bf16[2560][1]cuda:0", arg103_1: "bf16[2560, 2560][2560, 1]cuda:0", arg104_1: "bf16[2560][1]cuda:0", arg105_1: "bf16[2560, 2560][2560, 1]cuda:0", arg106_1: "bf16[2560][1]cuda:0", arg107_1: "bf16[2560, 2560][2560, 1]cuda:0", arg108_1: "bf16[2560][1]cuda:0", arg109_1: "bf16[2560][1]cuda:0", arg110_1: "bf16[2560][1]cuda:0", arg111_1: "bf16[10240, 2560][2560, 1]cuda:0", arg112_1: "bf16[10240][1]cuda:0", arg113_1: "bf16[2560, 10240][10240, 1]cuda:0", arg114_1: "bf16[2560][1]cuda:0", arg115_1: "bf16[2560][1]cuda:0", arg116_1: "bf16[2560][1]cuda:0", arg117_1: "bf16[2560, 2560][2560, 1]cuda:0", arg118_1: "bf16[2560][1]cuda:0", arg119_1: "bf16[2560, 2560][2560, 1]cuda:0", arg120_1: "bf16[2560][1]cuda:0", arg121_1: "bf16[2560, 2560][2560, 1]cuda:0", arg122_1: "bf16[2560][1]cuda:0", arg123_1: "bf16[2560, 2560][2560, 1]cuda:0", arg124_1: "bf16[2560][1]cuda:0", arg125_1: "bf16[2560][1]cuda:0", arg126_1: "bf16[2560][1]cuda:0", arg127_1: "bf16[10240, 2560][2560, 1]cuda:0", arg128_1: "bf16[10240][1]cuda:0", arg129_1: "bf16[2560, 10240][10240, 1]cuda:0", arg130_1: "bf16[2560][1]cuda:0", arg131_1: "bf16[2560][1]cuda:0", arg132_1: "bf16[2560][1]cuda:0", arg133_1: "bf16[2560, 2560][2560, 1]cuda:0", arg134_1: "bf16[2560][1]cuda:0", arg135_1: "bf16[2560, 2560][2560, 1]cuda:0", arg136_1: "bf16[2560][1]cuda:0", arg137_1: "bf16[2560, 2560][2560, 1]cuda:0", arg138_1: "bf16[2560][1]cuda:0", arg139_1: "bf16[2560, 2560][2560, 1]cuda:0", arg140_1: "bf16[2560][1]cuda:0", arg141_1: "bf16[2560][1]cuda:0", arg142_1: "bf16[2560][1]cuda:0", arg143_1: "bf16[10240, 2560][2560, 1]cuda:0", arg144_1: "bf16[10240][1]cuda:0", arg145_1: "bf16[2560, 10240][10240, 1]cuda:0", arg146_1: "bf16[2560][1]cuda:0", arg147_1: "bf16[2560][1]cuda:0", arg148_1: "bf16[2560][1]cuda:0", arg149_1: "bf16[2560, 2560][2560, 1]cuda:0", arg150_1: "bf16[2560][1]cuda:0", arg151_1: "bf16[2560, 2560][2560, 1]cuda:0", arg152_1: "bf16[2560][1]cuda:0", arg153_1: "bf16[2560, 2560][2560, 1]cuda:0", arg154_1: "bf16[2560][1]cuda:0", arg155_1: "bf16[2560, 2560][2560, 1]cuda:0", arg156_1: "bf16[2560][1]cuda:0", arg157_1: "bf16[2560][1]cuda:0", arg158_1: "bf16[2560][1]cuda:0", arg159_1: "bf16[10240, 2560][2560, 1]cuda:0", arg160_1: "bf16[10240][1]cuda:0", arg161_1: "bf16[2560, 10240][10240, 1]cuda:0", arg162_1: "bf16[2560][1]cuda:0", arg163_1: "bf16[2560][1]cuda:0", arg164_1: "bf16[2560][1]cuda:0", arg165_1: "bf16[2560, 2560][2560, 1]cuda:0", arg166_1: "bf16[2560][1]cuda:0", arg167_1: "bf16[2560, 2560][2560, 1]cuda:0", arg168_1: "bf16[2560][1]cuda:0", arg169_1: "bf16[2560, 2560][2560, 1]cuda:0", arg170_1: "bf16[2560][1]cuda:0", arg171_1: "bf16[2560, 2560][2560, 1]cuda:0", arg172_1: "bf16[2560][1]cuda:0", arg173_1: "bf16[2560][1]cuda:0", arg174_1: "bf16[2560][1]cuda:0", arg175_1: "bf16[10240, 2560][2560, 1]cuda:0", arg176_1: "bf16[10240][1]cuda:0", arg177_1: "bf16[2560, 10240][10240, 1]cuda:0", arg178_1: "bf16[2560][1]cuda:0", arg179_1: "bf16[2560][1]cuda:0", arg180_1: "bf16[2560][1]cuda:0", arg181_1: "bf16[2560, 2560][2560, 1]cuda:0", arg182_1: "bf16[2560][1]cuda:0", arg183_1: "bf16[2560, 2560][2560, 1]cuda:0", arg184_1: "bf16[2560][1]cuda:0", arg185_1: "bf16[2560, 2560][2560, 1]cuda:0", arg186_1: "bf16[2560][1]cuda:0", arg187_1: "bf16[2560, 2560][2560, 1]cuda:0", arg188_1: "bf16[2560][1]cuda:0", arg189_1: "bf16[2560][1]cuda:0", arg190_1: "bf16[2560][1]cuda:0", arg191_1: "bf16[10240, 2560][2560, 1]cuda:0", arg192_1: "bf16[10240][1]cuda:0", arg193_1: "bf16[2560, 10240][10240, 1]cuda:0", arg194_1: "bf16[2560][1]cuda:0", arg195_1: "bf16[2560][1]cuda:0", arg196_1: "bf16[2560][1]cuda:0", arg197_1: "bf16[2560, 2560][2560, 1]cuda:0", arg198_1: "bf16[2560][1]cuda:0", arg199_1: "bf16[2560, 2560][2560, 1]cuda:0", arg200_1: "bf16[2560][1]cuda:0", arg201_1: "bf16[2560, 2560][2560, 1]cuda:0", arg202_1: "bf16[2560][1]cuda:0", arg203_1: "bf16[2560, 2560][2560, 1]cuda:0", arg204_1: "bf16[2560][1]cuda:0", arg205_1: "bf16[2560][1]cuda:0", arg206_1: "bf16[2560][1]cuda:0", arg207_1: "bf16[10240, 2560][2560, 1]cuda:0", arg208_1: "bf16[10240][1]cuda:0", arg209_1: "bf16[2560, 10240][10240, 1]cuda:0", arg210_1: "bf16[2560][1]cuda:0", arg211_1: "bf16[2560][1]cuda:0", arg212_1: "bf16[2560][1]cuda:0", arg213_1: "bf16[2560, 2560][2560, 1]cuda:0", arg214_1: "bf16[2560][1]cuda:0", arg215_1: "bf16[2560, 2560][2560, 1]cuda:0", arg216_1: "bf16[2560][1]cuda:0", arg217_1: "bf16[2560, 2560][2560, 1]cuda:0", arg218_1: "bf16[2560][1]cuda:0", arg219_1: "bf16[2560, 2560][2560, 1]cuda:0", arg220_1: "bf16[2560][1]cuda:0", arg221_1: "bf16[2560][1]cuda:0", arg222_1: "bf16[2560][1]cuda:0", arg223_1: "bf16[10240, 2560][2560, 1]cuda:0", arg224_1: "bf16[10240][1]cuda:0", arg225_1: "bf16[2560, 10240][10240, 1]cuda:0", arg226_1: "bf16[2560][1]cuda:0", arg227_1: "bf16[2560][1]cuda:0", arg228_1: "bf16[2560][1]cuda:0", arg229_1: "bf16[2560, 2560][2560, 1]cuda:0", arg230_1: "bf16[2560][1]cuda:0", arg231_1: "bf16[2560, 2560][2560, 1]cuda:0", arg232_1: "bf16[2560][1]cuda:0", arg233_1: "bf16[2560, 2560][2560, 1]cuda:0", arg234_1: "bf16[2560][1]cuda:0", arg235_1: "bf16[2560, 2560][2560, 1]cuda:0", arg236_1: "bf16[2560][1]cuda:0", arg237_1: "bf16[2560][1]cuda:0", arg238_1: "bf16[2560][1]cuda:0", arg239_1: "bf16[10240, 2560][2560, 1]cuda:0", arg240_1: "bf16[10240][1]cuda:0", arg241_1: "bf16[2560, 10240][10240, 1]cuda:0", arg242_1: "bf16[2560][1]cuda:0", arg243_1: "bf16[2560][1]cuda:0", arg244_1: "bf16[2560][1]cuda:0", arg245_1: "bf16[2560, 2560][2560, 1]cuda:0", arg246_1: "bf16[2560][1]cuda:0", arg247_1: "bf16[2560, 2560][2560, 1]cuda:0", arg248_1: "bf16[2560][1]cuda:0", arg249_1: "bf16[2560, 2560][2560, 1]cuda:0", arg250_1: "bf16[2560][1]cuda:0", arg251_1: "bf16[2560, 2560][2560, 1]cuda:0", arg252_1: "bf16[2560][1]cuda:0", arg253_1: "bf16[2560][1]cuda:0", arg254_1: "bf16[2560][1]cuda:0", arg255_1: "bf16[10240, 2560][2560, 1]cuda:0", arg256_1: "bf16[10240][1]cuda:0", arg257_1: "bf16[2560, 10240][10240, 1]cuda:0", arg258_1: "bf16[2560][1]cuda:0", arg259_1: "bf16[2560][1]cuda:0", arg260_1: "bf16[2560][1]cuda:0", arg261_1: "bf16[2560, 2560][2560, 1]cuda:0", arg262_1: "bf16[2560][1]cuda:0", arg263_1: "bf16[2560, 2560][2560, 1]cuda:0", arg264_1: "bf16[2560][1]cuda:0", arg265_1: "bf16[2560, 2560][2560, 1]cuda:0", arg266_1: "bf16[2560][1]cuda:0", arg267_1: "bf16[2560, 2560][2560, 1]cuda:0", arg268_1: "bf16[2560][1]cuda:0", arg269_1: "bf16[2560][1]cuda:0", arg270_1: "bf16[2560][1]cuda:0", arg271_1: "bf16[10240, 2560][2560, 1]cuda:0", arg272_1: "bf16[10240][1]cuda:0", arg273_1: "bf16[2560, 10240][10240, 1]cuda:0", arg274_1: "bf16[2560][1]cuda:0", arg275_1: "bf16[2560][1]cuda:0", arg276_1: "bf16[2560][1]cuda:0", arg277_1: "bf16[2560, 2560][2560, 1]cuda:0", arg278_1: "bf16[2560][1]cuda:0", arg279_1: "bf16[2560, 2560][2560, 1]cuda:0", arg280_1: "bf16[2560][1]cuda:0", arg281_1: "bf16[2560, 2560][2560, 1]cuda:0", arg282_1: "bf16[2560][1]cuda:0", arg283_1: "bf16[2560, 2560][2560, 1]cuda:0", arg284_1: "bf16[2560][1]cuda:0", arg285_1: "bf16[2560][1]cuda:0", arg286_1: "bf16[2560][1]cuda:0", arg287_1: "bf16[10240, 2560][2560, 1]cuda:0", arg288_1: "bf16[10240][1]cuda:0", arg289_1: "bf16[2560, 10240][10240, 1]cuda:0", arg290_1: "bf16[2560][1]cuda:0", arg291_1: "bf16[2560][1]cuda:0", arg292_1: "bf16[2560][1]cuda:0", arg293_1: "bf16[2560, 2560][2560, 1]cuda:0", arg294_1: "bf16[2560][1]cuda:0", arg295_1: "bf16[2560, 2560][2560, 1]cuda:0", arg296_1: "bf16[2560][1]cuda:0", arg297_1: "bf16[2560, 2560][2560, 1]cuda:0", arg298_1: "bf16[2560][1]cuda:0", arg299_1: "bf16[2560, 2560][2560, 1]cuda:0", arg300_1: "bf16[2560][1]cuda:0", arg301_1: "bf16[2560][1]cuda:0", arg302_1: "bf16[2560][1]cuda:0", arg303_1: "bf16[10240, 2560][2560, 1]cuda:0", arg304_1: "bf16[10240][1]cuda:0", arg305_1: "bf16[2560, 10240][10240, 1]cuda:0", arg306_1: "bf16[2560][1]cuda:0", arg307_1: "bf16[2560][1]cuda:0", arg308_1: "bf16[2560][1]cuda:0", arg309_1: "bf16[2560, 2560][2560, 1]cuda:0", arg310_1: "bf16[2560][1]cuda:0", arg311_1: "bf16[2560, 2560][2560, 1]cuda:0", arg312_1: "bf16[2560][1]cuda:0", arg313_1: "bf16[2560, 2560][2560, 1]cuda:0", arg314_1: "bf16[2560][1]cuda:0", arg315_1: "bf16[2560, 2560][2560, 1]cuda:0", arg316_1: "bf16[2560][1]cuda:0", arg317_1: "bf16[2560][1]cuda:0", arg318_1: "bf16[2560][1]cuda:0", arg319_1: "bf16[10240, 2560][2560, 1]cuda:0", arg320_1: "bf16[10240][1]cuda:0", arg321_1: "bf16[2560, 10240][10240, 1]cuda:0", arg322_1: "bf16[2560][1]cuda:0", arg323_1: "bf16[2560][1]cuda:0", arg324_1: "bf16[2560][1]cuda:0", arg325_1: "bf16[2560, 2560][2560, 1]cuda:0", arg326_1: "bf16[2560][1]cuda:0", arg327_1: "bf16[2560, 2560][2560, 1]cuda:0", arg328_1: "bf16[2560][1]cuda:0", arg329_1: "bf16[2560, 2560][2560, 1]cuda:0", arg330_1: "bf16[2560][1]cuda:0", arg331_1: "bf16[2560, 2560][2560, 1]cuda:0", arg332_1: "bf16[2560][1]cuda:0", arg333_1: "bf16[2560][1]cuda:0", arg334_1: "bf16[2560][1]cuda:0", arg335_1: "bf16[10240, 2560][2560, 1]cuda:0", arg336_1: "bf16[10240][1]cuda:0", arg337_1: "bf16[2560, 10240][10240, 1]cuda:0", arg338_1: "bf16[2560][1]cuda:0", arg339_1: "bf16[2560][1]cuda:0", arg340_1: "bf16[2560][1]cuda:0", arg341_1: "bf16[2560, 2560][2560, 1]cuda:0", arg342_1: "bf16[2560][1]cuda:0", arg343_1: "bf16[2560, 2560][2560, 1]cuda:0", arg344_1: "bf16[2560][1]cuda:0", arg345_1: "bf16[2560, 2560][2560, 1]cuda:0", arg346_1: "bf16[2560][1]cuda:0", arg347_1: "bf16[2560, 2560][2560, 1]cuda:0", arg348_1: "bf16[2560][1]cuda:0", arg349_1: "bf16[2560][1]cuda:0", arg350_1: "bf16[2560][1]cuda:0", arg351_1: "bf16[10240, 2560][2560, 1]cuda:0", arg352_1: "bf16[10240][1]cuda:0", arg353_1: "bf16[2560, 10240][10240, 1]cuda:0", arg354_1: "bf16[2560][1]cuda:0", arg355_1: "bf16[2560][1]cuda:0", arg356_1: "bf16[2560][1]cuda:0", arg357_1: "bf16[2560, 2560][2560, 1]cuda:0", arg358_1: "bf16[2560][1]cuda:0", arg359_1: "bf16[2560, 2560][2560, 1]cuda:0", arg360_1: "bf16[2560][1]cuda:0", arg361_1: "bf16[2560, 2560][2560, 1]cuda:0", arg362_1: "bf16[2560][1]cuda:0", arg363_1: "bf16[2560, 2560][2560, 1]cuda:0", arg364_1: "bf16[2560][1]cuda:0", arg365_1: "bf16[2560][1]cuda:0", arg366_1: "bf16[2560][1]cuda:0", arg367_1: "bf16[10240, 2560][2560, 1]cuda:0", arg368_1: "bf16[10240][1]cuda:0", arg369_1: "bf16[2560, 10240][10240, 1]cuda:0", arg370_1: "bf16[2560][1]cuda:0", arg371_1: "bf16[2560][1]cuda:0", arg372_1: "bf16[2560][1]cuda:0", arg373_1: "bf16[2560, 2560][2560, 1]cuda:0", arg374_1: "bf16[2560][1]cuda:0", arg375_1: "bf16[2560, 2560][2560, 1]cuda:0", arg376_1: "bf16[2560][1]cuda:0", arg377_1: "bf16[2560, 2560][2560, 1]cuda:0", arg378_1: "bf16[2560][1]cuda:0", arg379_1: "bf16[2560, 2560][2560, 1]cuda:0", arg380_1: "bf16[2560][1]cuda:0", arg381_1: "bf16[2560][1]cuda:0", arg382_1: "bf16[2560][1]cuda:0", arg383_1: "bf16[10240, 2560][2560, 1]cuda:0", arg384_1: "bf16[10240][1]cuda:0", arg385_1: "bf16[2560, 10240][10240, 1]cuda:0", arg386_1: "bf16[2560][1]cuda:0", arg387_1: "bf16[2560][1]cuda:0", arg388_1: "bf16[2560][1]cuda:0", arg389_1: "i64[32, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:96 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None
        mul: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:585 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_1: "bf16[128, 2560][2560, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, add);  arg2_1 = add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:616 in forward, code: hidden_states = inputs_embeds + position_ids
        add_5: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_1: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_7: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 2560])
        permute: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
        view_1: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [32, 128, 2560]);  addmm = None
        view_2: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [32, 128, -1, 80]);  view_1 = None
        permute_1: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_3: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 2560])
        permute_2: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg8_1, view_3, permute_2);  arg8_1 = view_3 = permute_2 = None
        view_4: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 128, 2560]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [32, 128, -1, 80]);  view_4 = None
        permute_4: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_5: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 2560]);  convert_element_type_1 = None
        permute_3: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg10_1, view_5, permute_3);  arg10_1 = view_5 = permute_3 = None
        view_6: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 128, 2560]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [32, 128, -1, 80]);  view_6 = None
        permute_5: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_3: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_4: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_1: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2);  unsqueeze_5 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[32, 1, 128, 128][0, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(le, [32, -1, 128, 128]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_1, full_default);  full_default_1 = full_default = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_1, permute_4, permute_5, where, False, scale = 0.11180339887498948);  permute_1 = permute_4 = permute_5 = where = None
        getitem_2: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [32, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [4096, 2560]);  view_9 = None
        permute_7: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view_10, permute_7);  arg12_1 = view_10 = permute_7 = None
        view_11: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [32, 128, 2560]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_8: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_5, view_11);  add_5 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_14: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_14, [2], correction = 0, keepdim = True)
        getitem_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_12);  convert_element_type_14 = getitem_12 = None
        add_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_3: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_4: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, arg13_1);  mul_3 = arg13_1 = None
        add_10: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
        convert_element_type_15: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_12: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_15, [4096, 2560]);  convert_element_type_15 = None
        permute_8: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, view_12, permute_8);  arg16_1 = view_12 = permute_8 = None
        view_13: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 10240]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_19: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_5: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, 0.5)
        mul_6: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, 0.7071067811865476);  convert_element_type_19 = None
        erf: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_11: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, add_11);  mul_5 = add_11 = None
        convert_element_type_20: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_14: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_20, [4096, 10240]);  convert_element_type_20 = None
        permute_9: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg18_1, view_14, permute_9);  arg18_1 = view_14 = permute_9 = None
        view_15: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 128, 2560]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_12: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_8, view_15);  add_8 = view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_24: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_2[0]
        getitem_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_14);  convert_element_type_24 = getitem_14 = None
        add_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_8: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, arg19_1);  mul_8 = arg19_1 = None
        add_14: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        convert_element_type_25: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_16: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [4096, 2560])
        permute_10: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg22_1, view_16, permute_10);  arg22_1 = view_16 = permute_10 = None
        view_17: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [32, 128, 2560]);  addmm_6 = None
        view_18: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_17, [32, 128, -1, 80]);  view_17 = None
        permute_11: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_19: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [4096, 2560])
        permute_12: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg24_1, view_19, permute_12);  arg24_1 = view_19 = permute_12 = None
        view_20: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 128, 2560]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_23: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_20, [32, 128, -1, 80]);  view_20 = None
        permute_14: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_21: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [4096, 2560]);  convert_element_type_25 = None
        permute_13: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg26_1, view_21, permute_13);  arg26_1 = view_21 = permute_13 = None
        view_22: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [32, 128, 2560]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_24: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_22, [32, 128, -1, 80]);  view_22 = None
        permute_15: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_3, full_default_2);  full_default_3 = full_default_2 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_11, permute_14, permute_15, where_1, False, scale = 0.11180339887498948);  permute_11 = permute_14 = permute_15 = where_1 = None
        getitem_15: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_16: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_16, [32, 128, -1]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_26: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [4096, 2560]);  view_25 = None
        permute_17: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg28_1, view_26, permute_17);  arg28_1 = view_26 = permute_17 = None
        view_27: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [32, 128, 2560]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_15: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_12, view_27);  add_12 = view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_38: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_38, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_3[0]
        getitem_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, getitem_25);  convert_element_type_38 = getitem_25 = None
        add_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_10: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, arg29_1);  mul_10 = arg29_1 = None
        add_17: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, arg30_1);  mul_11 = arg30_1 = None
        convert_element_type_39: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_28: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_39, [4096, 2560]);  convert_element_type_39 = None
        permute_18: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, view_28, permute_18);  arg32_1 = view_28 = permute_18 = None
        view_29: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 128, 10240]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_43: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        mul_12: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.5)
        mul_13: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.7071067811865476);  convert_element_type_43 = None
        erf_1: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_18: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_18);  mul_12 = add_18 = None
        convert_element_type_44: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_30: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [4096, 10240]);  convert_element_type_44 = None
        permute_19: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg34_1, view_30, permute_19);  arg34_1 = view_30 = permute_19 = None
        view_31: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [32, 128, 2560]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_19: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_15, view_31);  add_15 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_48: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_4[0]
        getitem_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_27);  convert_element_type_48 = getitem_27 = None
        add_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_15: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_16: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, arg35_1);  mul_15 = arg35_1 = None
        add_21: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, arg36_1);  mul_16 = arg36_1 = None
        convert_element_type_49: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_32: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [4096, 2560])
        permute_20: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg38_1, view_32, permute_20);  arg38_1 = view_32 = permute_20 = None
        view_33: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [32, 128, 2560]);  addmm_12 = None
        view_34: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_33, [32, 128, -1, 80]);  view_33 = None
        permute_21: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_35: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [4096, 2560])
        permute_22: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg40_1, view_35, permute_22);  arg40_1 = view_35 = permute_22 = None
        view_36: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 128, 2560]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_39: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_36, [32, 128, -1, 80]);  view_36 = None
        permute_24: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_37: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [4096, 2560]);  convert_element_type_49 = None
        permute_23: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_37, permute_23);  arg42_1 = view_37 = permute_23 = None
        view_38: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [32, 128, 2560]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_40: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_38, [32, 128, -1, 80]);  view_38 = None
        permute_25: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_5: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_5, full_default_4);  full_default_5 = full_default_4 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_21, permute_24, permute_25, where_2, False, scale = 0.11180339887498948);  permute_21 = permute_24 = permute_25 = where_2 = None
        getitem_28: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_26: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_41: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [32, 128, -1]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_42: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_41, [4096, 2560]);  view_41 = None
        permute_27: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_42, permute_27);  arg44_1 = view_42 = permute_27 = None
        view_43: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [32, 128, 2560]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_22: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_19, view_43);  add_19 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_62: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_62, [2], correction = 0, keepdim = True)
        getitem_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_5[0]
        getitem_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_38);  convert_element_type_62 = getitem_38 = None
        add_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_17: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_18: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, arg45_1);  mul_17 = arg45_1 = None
        add_24: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, arg46_1);  mul_18 = arg46_1 = None
        convert_element_type_63: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_44: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [4096, 2560]);  convert_element_type_63 = None
        permute_28: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, view_44, permute_28);  arg48_1 = view_44 = permute_28 = None
        view_45: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 128, 10240]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_67: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_19: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_20: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476);  convert_element_type_67 = None
        erf_2: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_25: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, add_25);  mul_19 = add_25 = None
        convert_element_type_68: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_46: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_68, [4096, 10240]);  convert_element_type_68 = None
        permute_29: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg50_1, view_46, permute_29);  arg50_1 = view_46 = permute_29 = None
        view_47: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [32, 128, 2560]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_26: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_22, view_47);  add_22 = view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_72: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_6[0]
        getitem_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_40);  convert_element_type_72 = getitem_40 = None
        add_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_22: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_23: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg51_1);  mul_22 = arg51_1 = None
        add_28: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg52_1);  mul_23 = arg52_1 = None
        convert_element_type_73: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_48: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [4096, 2560])
        permute_30: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg54_1, view_48, permute_30);  arg54_1 = view_48 = permute_30 = None
        view_49: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [32, 128, 2560]);  addmm_18 = None
        view_50: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_49, [32, 128, -1, 80]);  view_49 = None
        permute_31: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_51: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [4096, 2560])
        permute_32: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg56_1, view_51, permute_32);  arg56_1 = view_51 = permute_32 = None
        view_52: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 128, 2560]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_55: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [32, 128, -1, 80]);  view_52 = None
        permute_34: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_53: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [4096, 2560]);  convert_element_type_73 = None
        permute_33: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg58_1, view_53, permute_33);  arg58_1 = view_53 = permute_33 = None
        view_54: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [32, 128, 2560]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_56: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [32, 128, -1, 80]);  view_54 = None
        permute_35: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_31, permute_34, permute_35, where_3, False, scale = 0.11180339887498948);  permute_31 = permute_34 = permute_35 = where_3 = None
        getitem_41: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_36: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_41, [0, 2, 1, 3]);  getitem_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_57: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_36, [32, 128, -1]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_58: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_57, [4096, 2560]);  view_57 = None
        permute_37: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg60_1, view_58, permute_37);  arg60_1 = view_58 = permute_37 = None
        view_59: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [32, 128, 2560]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_29: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_26, view_59);  add_26 = view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_86: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_86, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_7[0]
        getitem_51: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_51);  convert_element_type_86 = getitem_51 = None
        add_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_24: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_25: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, arg61_1);  mul_24 = arg61_1 = None
        add_31: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, arg62_1);  mul_25 = arg62_1 = None
        convert_element_type_87: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_60: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_87, [4096, 2560]);  convert_element_type_87 = None
        permute_38: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg64_1, view_60, permute_38);  arg64_1 = view_60 = permute_38 = None
        view_61: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 128, 10240]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_91: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        mul_26: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5)
        mul_27: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476);  convert_element_type_91 = None
        erf_3: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_32: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
        convert_element_type_92: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_62: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [4096, 10240]);  convert_element_type_92 = None
        permute_39: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg66_1, view_62, permute_39);  arg66_1 = view_62 = permute_39 = None
        view_63: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [32, 128, 2560]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_33: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_29, view_63);  add_29 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_96: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_52: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_8[0]
        getitem_53: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_53);  convert_element_type_96 = getitem_53 = None
        add_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_29: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_30: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, arg67_1);  mul_29 = arg67_1 = None
        add_35: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, arg68_1);  mul_30 = arg68_1 = None
        convert_element_type_97: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_64: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [4096, 2560])
        permute_40: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg70_1, view_64, permute_40);  arg70_1 = view_64 = permute_40 = None
        view_65: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [32, 128, 2560]);  addmm_24 = None
        view_66: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_65, [32, 128, -1, 80]);  view_65 = None
        permute_41: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_67: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [4096, 2560])
        permute_42: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_67, permute_42);  arg72_1 = view_67 = permute_42 = None
        view_68: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 128, 2560]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_71: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_68, [32, 128, -1, 80]);  view_68 = None
        permute_44: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_69: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [4096, 2560]);  convert_element_type_97 = None
        permute_43: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_69, permute_43);  arg74_1 = view_69 = permute_43 = None
        view_70: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [32, 128, 2560]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_72: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [32, 128, -1, 80]);  view_70 = None
        permute_45: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_41, permute_44, permute_45, where_4, False, scale = 0.11180339887498948);  permute_41 = permute_44 = permute_45 = where_4 = None
        getitem_54: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_46: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_73: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_46, [32, 128, -1]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_74: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [4096, 2560]);  view_73 = None
        permute_47: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg76_1, view_74, permute_47);  arg76_1 = view_74 = permute_47 = None
        view_75: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [32, 128, 2560]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_36: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_33, view_75);  add_33 = view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_110: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_63: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_9[0]
        getitem_64: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_64);  convert_element_type_110 = getitem_64 = None
        add_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_31: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_32: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, arg77_1);  mul_31 = arg77_1 = None
        add_38: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, arg78_1);  mul_32 = arg78_1 = None
        convert_element_type_111: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_76: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_111, [4096, 2560]);  convert_element_type_111 = None
        permute_48: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_76, permute_48);  arg80_1 = view_76 = permute_48 = None
        view_77: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 128, 10240]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_115: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_33: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5)
        mul_34: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.7071067811865476);  convert_element_type_115 = None
        erf_4: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_39: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, add_39);  mul_33 = add_39 = None
        convert_element_type_116: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_78: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [4096, 10240]);  convert_element_type_116 = None
        permute_49: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg82_1, view_78, permute_49);  arg82_1 = view_78 = permute_49 = None
        view_79: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [32, 128, 2560]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_40: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_36, view_79);  add_36 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_120: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_65: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_10[0]
        getitem_66: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_66);  convert_element_type_120 = getitem_66 = None
        add_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_36: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_37: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg83_1);  mul_36 = arg83_1 = None
        add_42: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg84_1);  mul_37 = arg84_1 = None
        convert_element_type_121: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_42, torch.bfloat16);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_80: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [4096, 2560])
        permute_50: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg86_1, view_80, permute_50);  arg86_1 = view_80 = permute_50 = None
        view_81: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [32, 128, 2560]);  addmm_30 = None
        view_82: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [32, 128, -1, 80]);  view_81 = None
        permute_51: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_83: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [4096, 2560])
        permute_52: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg88_1, view_83, permute_52);  arg88_1 = view_83 = permute_52 = None
        view_84: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 128, 2560]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_87: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [32, 128, -1, 80]);  view_84 = None
        permute_54: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_85: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [4096, 2560]);  convert_element_type_121 = None
        permute_53: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg90_1, view_85, permute_53);  arg90_1 = view_85 = permute_53 = None
        view_86: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [32, 128, 2560]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_88: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [32, 128, -1, 80]);  view_86 = None
        permute_55: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_11: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_11, full_default_10);  full_default_11 = full_default_10 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_51, permute_54, permute_55, where_5, False, scale = 0.11180339887498948);  permute_51 = permute_54 = permute_55 = where_5 = None
        getitem_67: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_67, [0, 2, 1, 3]);  getitem_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_89: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_56, [32, 128, -1]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_90: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [4096, 2560]);  view_89 = None
        permute_57: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg92_1, view_90, permute_57);  arg92_1 = view_90 = permute_57 = None
        view_91: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [32, 128, 2560]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_43: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_40, view_91);  add_40 = view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_134: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_134, [2], correction = 0, keepdim = True)
        getitem_76: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_11[0]
        getitem_77: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_77);  convert_element_type_134 = getitem_77 = None
        add_44: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_38: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_39: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, arg93_1);  mul_38 = arg93_1 = None
        add_45: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, arg94_1);  mul_39 = arg94_1 = None
        convert_element_type_135: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_92: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [4096, 2560]);  convert_element_type_135 = None
        permute_58: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg96_1, view_92, permute_58);  arg96_1 = view_92 = permute_58 = None
        view_93: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 128, 10240]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_139: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_40: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_41: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_5: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_46: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, add_46);  mul_40 = add_46 = None
        convert_element_type_140: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_94: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_140, [4096, 10240]);  convert_element_type_140 = None
        permute_59: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, view_94, permute_59);  arg98_1 = view_94 = permute_59 = None
        view_95: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [32, 128, 2560]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_47: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_43, view_95);  add_43 = view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_144: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_12[0]
        getitem_79: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_12: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_79);  convert_element_type_144 = getitem_79 = None
        add_48: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_43: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_44: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, arg99_1);  mul_43 = arg99_1 = None
        add_49: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, arg100_1);  mul_44 = arg100_1 = None
        convert_element_type_145: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_96: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [4096, 2560])
        permute_60: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg102_1, view_96, permute_60);  arg102_1 = view_96 = permute_60 = None
        view_97: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [32, 128, 2560]);  addmm_36 = None
        view_98: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_97, [32, 128, -1, 80]);  view_97 = None
        permute_61: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_99: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [4096, 2560])
        permute_62: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg104_1, view_99, permute_62);  arg104_1 = view_99 = permute_62 = None
        view_100: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 128, 2560]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_103: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_100, [32, 128, -1, 80]);  view_100 = None
        permute_64: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_101: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [4096, 2560]);  convert_element_type_145 = None
        permute_63: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg106_1, view_101, permute_63);  arg106_1 = view_101 = permute_63 = None
        view_102: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [32, 128, 2560]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_104: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [32, 128, -1, 80]);  view_102 = None
        permute_65: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_61, permute_64, permute_65, where_6, False, scale = 0.11180339887498948);  permute_61 = permute_64 = permute_65 = where_6 = None
        getitem_80: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_66: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3]);  getitem_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_66, [32, 128, -1]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_106: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [4096, 2560]);  view_105 = None
        permute_67: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg108_1, view_106, permute_67);  arg108_1 = view_106 = permute_67 = None
        view_107: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [32, 128, 2560]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_50: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_47, view_107);  add_47 = view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_158: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_158, [2], correction = 0, keepdim = True)
        getitem_89: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_13[0]
        getitem_90: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_158, getitem_90);  convert_element_type_158 = getitem_90 = None
        add_51: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
        rsqrt_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_45: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_46: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, arg109_1);  mul_45 = arg109_1 = None
        add_52: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, arg110_1);  mul_46 = arg110_1 = None
        convert_element_type_159: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_108: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_159, [4096, 2560]);  convert_element_type_159 = None
        permute_68: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg112_1, view_108, permute_68);  arg112_1 = view_108 = permute_68 = None
        view_109: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 128, 10240]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_163: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        mul_47: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_163, 0.5)
        mul_48: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_163, 0.7071067811865476);  convert_element_type_163 = None
        erf_6: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_53: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, add_53);  mul_47 = add_53 = None
        convert_element_type_164: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_110: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_164, [4096, 10240]);  convert_element_type_164 = None
        permute_69: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg114_1, view_110, permute_69);  arg114_1 = view_110 = permute_69 = None
        view_111: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [32, 128, 2560]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_54: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_50, view_111);  add_50 = view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_168: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_168, [2], correction = 0, keepdim = True)
        getitem_91: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_14[0]
        getitem_92: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_168, getitem_92);  convert_element_type_168 = getitem_92 = None
        add_55: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_50: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_51: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, arg115_1);  mul_50 = arg115_1 = None
        add_56: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, arg116_1);  mul_51 = arg116_1 = None
        convert_element_type_169: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_112: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_169, [4096, 2560])
        permute_70: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg118_1, view_112, permute_70);  arg118_1 = view_112 = permute_70 = None
        view_113: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [32, 128, 2560]);  addmm_42 = None
        view_114: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [32, 128, -1, 80]);  view_113 = None
        permute_71: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_115: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_169, [4096, 2560])
        permute_72: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg120_1, view_115, permute_72);  arg120_1 = view_115 = permute_72 = None
        view_116: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [32, 128, 2560]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_119: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_116, [32, 128, -1, 80]);  view_116 = None
        permute_74: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_117: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_169, [4096, 2560]);  convert_element_type_169 = None
        permute_73: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg122_1, view_117, permute_73);  arg122_1 = view_117 = permute_73 = None
        view_118: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [32, 128, 2560]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_120: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_118, [32, 128, -1, 80]);  view_118 = None
        permute_75: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_15: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_15, full_default_14);  full_default_15 = full_default_14 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_71, permute_74, permute_75, where_7, False, scale = 0.11180339887498948);  permute_71 = permute_74 = permute_75 = where_7 = None
        getitem_93: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_76: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_121: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_76, [32, 128, -1]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_122: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_121, [4096, 2560]);  view_121 = None
        permute_77: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg124_1, view_122, permute_77);  arg124_1 = view_122 = permute_77 = None
        view_123: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [32, 128, 2560]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_57: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_54, view_123);  add_54 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_182: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_182, [2], correction = 0, keepdim = True)
        getitem_102: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_15[0]
        getitem_103: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_182, getitem_103);  convert_element_type_182 = getitem_103 = None
        add_58: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_15: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_52: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_53: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, arg125_1);  mul_52 = arg125_1 = None
        add_59: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, arg126_1);  mul_53 = arg126_1 = None
        convert_element_type_183: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_124: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_183, [4096, 2560]);  convert_element_type_183 = None
        permute_78: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg128_1, view_124, permute_78);  arg128_1 = view_124 = permute_78 = None
        view_125: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 128, 10240]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_187: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        mul_54: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.5)
        mul_55: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.7071067811865476);  convert_element_type_187 = None
        erf_7: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_60: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, add_60);  mul_54 = add_60 = None
        convert_element_type_188: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_126: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_188, [4096, 10240]);  convert_element_type_188 = None
        permute_79: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg130_1, view_126, permute_79);  arg130_1 = view_126 = permute_79 = None
        view_127: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [32, 128, 2560]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_61: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, view_127);  add_57 = view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_192: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_192, [2], correction = 0, keepdim = True)
        getitem_104: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_16[0]
        getitem_105: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_16: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_105);  convert_element_type_192 = getitem_105 = None
        add_62: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_57: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_58: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg131_1);  mul_57 = arg131_1 = None
        add_63: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg132_1);  mul_58 = arg132_1 = None
        convert_element_type_193: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_128: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [4096, 2560])
        permute_80: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg134_1, view_128, permute_80);  arg134_1 = view_128 = permute_80 = None
        view_129: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [32, 128, 2560]);  addmm_48 = None
        view_130: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_129, [32, 128, -1, 80]);  view_129 = None
        permute_81: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_131: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [4096, 2560])
        permute_82: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg136_1, view_131, permute_82);  arg136_1 = view_131 = permute_82 = None
        view_132: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [32, 128, 2560]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_135: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_132, [32, 128, -1, 80]);  view_132 = None
        permute_84: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_133: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [4096, 2560]);  convert_element_type_193 = None
        permute_83: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg138_1, view_133, permute_83);  arg138_1 = view_133 = permute_83 = None
        view_134: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [32, 128, 2560]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_136: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_134, [32, 128, -1, 80]);  view_134 = None
        permute_85: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_81, permute_84, permute_85, where_8, False, scale = 0.11180339887498948);  permute_81 = permute_84 = permute_85 = where_8 = None
        getitem_106: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_86: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_137: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_86, [32, 128, -1]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_138: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [4096, 2560]);  view_137 = None
        permute_87: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_138, permute_87);  arg140_1 = view_138 = permute_87 = None
        view_139: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [32, 128, 2560]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_64: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_61, view_139);  add_61 = view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_206: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_206, [2], correction = 0, keepdim = True)
        getitem_115: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_17[0]
        getitem_116: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_206, getitem_116);  convert_element_type_206 = getitem_116 = None
        add_65: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_59: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_60: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, arg141_1);  mul_59 = arg141_1 = None
        add_66: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, arg142_1);  mul_60 = arg142_1 = None
        convert_element_type_207: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_140: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_207, [4096, 2560]);  convert_element_type_207 = None
        permute_88: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg144_1, view_140, permute_88);  arg144_1 = view_140 = permute_88 = None
        view_141: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 128, 10240]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_211: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_61: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.5)
        mul_62: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.7071067811865476);  convert_element_type_211 = None
        erf_8: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_67: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, add_67);  mul_61 = add_67 = None
        convert_element_type_212: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_142: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_212, [4096, 10240]);  convert_element_type_212 = None
        permute_89: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg146_1, view_142, permute_89);  arg146_1 = view_142 = permute_89 = None
        view_143: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [32, 128, 2560]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_68: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_64, view_143);  add_64 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_216: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_216, [2], correction = 0, keepdim = True)
        getitem_117: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_18[0]
        getitem_118: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_216, getitem_118);  convert_element_type_216 = getitem_118 = None
        add_69: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_18: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_64: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_65: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, arg147_1);  mul_64 = arg147_1 = None
        add_70: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, arg148_1);  mul_65 = arg148_1 = None
        convert_element_type_217: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_144: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_217, [4096, 2560])
        permute_90: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg150_1, view_144, permute_90);  arg150_1 = view_144 = permute_90 = None
        view_145: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [32, 128, 2560]);  addmm_54 = None
        view_146: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_145, [32, 128, -1, 80]);  view_145 = None
        permute_91: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_147: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_217, [4096, 2560])
        permute_92: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg152_1, view_147, permute_92);  arg152_1 = view_147 = permute_92 = None
        view_148: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [32, 128, 2560]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_151: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_148, [32, 128, -1, 80]);  view_148 = None
        permute_94: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_149: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_217, [4096, 2560]);  convert_element_type_217 = None
        permute_93: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg154_1, view_149, permute_93);  arg154_1 = view_149 = permute_93 = None
        view_150: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [32, 128, 2560]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_152: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_150, [32, 128, -1, 80]);  view_150 = None
        permute_95: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_19: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_91, permute_94, permute_95, where_9, False, scale = 0.11180339887498948);  permute_91 = permute_94 = permute_95 = where_9 = None
        getitem_119: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_153: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_96, [32, 128, -1]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_154: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_153, [4096, 2560]);  view_153 = None
        permute_97: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg156_1, view_154, permute_97);  arg156_1 = view_154 = permute_97 = None
        view_155: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [32, 128, 2560]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_71: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_68, view_155);  add_68 = view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_230: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_230, [2], correction = 0, keepdim = True)
        getitem_128: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_19[0]
        getitem_129: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, getitem_129);  convert_element_type_230 = getitem_129 = None
        add_72: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_19: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_66: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_67: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg157_1);  mul_66 = arg157_1 = None
        add_73: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg158_1);  mul_67 = arg158_1 = None
        convert_element_type_231: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_156: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_231, [4096, 2560]);  convert_element_type_231 = None
        permute_98: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_156, permute_98);  arg160_1 = view_156 = permute_98 = None
        view_157: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 128, 10240]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_235: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        mul_68: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.5)
        mul_69: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.7071067811865476);  convert_element_type_235 = None
        erf_9: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_74: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_74);  mul_68 = add_74 = None
        convert_element_type_236: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_158: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [4096, 10240]);  convert_element_type_236 = None
        permute_99: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg162_1, view_158, permute_99);  arg162_1 = view_158 = permute_99 = None
        view_159: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [32, 128, 2560]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_75: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_71, view_159);  add_71 = view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_240: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_240, [2], correction = 0, keepdim = True)
        getitem_130: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_20[0]
        getitem_131: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_20: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, getitem_131);  convert_element_type_240 = getitem_131 = None
        add_76: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_71: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_72: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, arg163_1);  mul_71 = arg163_1 = None
        add_77: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_72, arg164_1);  mul_72 = arg164_1 = None
        convert_element_type_241: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_160: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [4096, 2560])
        permute_100: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg166_1, view_160, permute_100);  arg166_1 = view_160 = permute_100 = None
        view_161: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [32, 128, 2560]);  addmm_60 = None
        view_162: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [32, 128, -1, 80]);  view_161 = None
        permute_101: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_163: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [4096, 2560])
        permute_102: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg168_1, view_163, permute_102);  arg168_1 = view_163 = permute_102 = None
        view_164: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 128, 2560]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_167: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_164, [32, 128, -1, 80]);  view_164 = None
        permute_104: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_165: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [4096, 2560]);  convert_element_type_241 = None
        permute_103: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg170_1, view_165, permute_103);  arg170_1 = view_165 = permute_103 = None
        view_166: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [32, 128, 2560]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_168: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_166, [32, 128, -1, 80]);  view_166 = None
        permute_105: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_21: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_21, full_default_20);  full_default_21 = full_default_20 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_101, permute_104, permute_105, where_10, False, scale = 0.11180339887498948);  permute_101 = permute_104 = permute_105 = where_10 = None
        getitem_132: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_106, [32, 128, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_170: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [4096, 2560]);  view_169 = None
        permute_107: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg172_1, view_170, permute_107);  arg172_1 = view_170 = permute_107 = None
        view_171: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [32, 128, 2560]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_78: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_75, view_171);  add_75 = view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_254: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_254, [2], correction = 0, keepdim = True)
        getitem_141: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_21[0]
        getitem_142: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_254, getitem_142);  convert_element_type_254 = getitem_142 = None
        add_79: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_73: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_74: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg173_1);  mul_73 = arg173_1 = None
        add_80: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg174_1);  mul_74 = arg174_1 = None
        convert_element_type_255: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_172: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_255, [4096, 2560]);  convert_element_type_255 = None
        permute_108: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg176_1, view_172, permute_108);  arg176_1 = view_172 = permute_108 = None
        view_173: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 128, 10240]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_259: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_75: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_259, 0.5)
        mul_76: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_259, 0.7071067811865476);  convert_element_type_259 = None
        erf_10: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_81: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, add_81);  mul_75 = add_81 = None
        convert_element_type_260: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_174: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_260, [4096, 10240]);  convert_element_type_260 = None
        permute_109: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg178_1, view_174, permute_109);  arg178_1 = view_174 = permute_109 = None
        view_175: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [32, 128, 2560]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_82: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_78, view_175);  add_78 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_264: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_82, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_264, [2], correction = 0, keepdim = True)
        getitem_143: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_22[0]
        getitem_144: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_22: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_264, getitem_144);  convert_element_type_264 = getitem_144 = None
        add_83: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_78: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_79: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, arg179_1);  mul_78 = arg179_1 = None
        add_84: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, arg180_1);  mul_79 = arg180_1 = None
        convert_element_type_265: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_176: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [4096, 2560])
        permute_110: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg182_1, view_176, permute_110);  arg182_1 = view_176 = permute_110 = None
        view_177: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [32, 128, 2560]);  addmm_66 = None
        view_178: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [32, 128, -1, 80]);  view_177 = None
        permute_111: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_179: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [4096, 2560])
        permute_112: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg184_1, view_179, permute_112);  arg184_1 = view_179 = permute_112 = None
        view_180: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [32, 128, 2560]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_183: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_180, [32, 128, -1, 80]);  view_180 = None
        permute_114: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_181: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [4096, 2560]);  convert_element_type_265 = None
        permute_113: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg186_1, view_181, permute_113);  arg186_1 = view_181 = permute_113 = None
        view_182: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [32, 128, 2560]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_184: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_182, [32, 128, -1, 80]);  view_182 = None
        permute_115: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_23: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_23, full_default_22);  full_default_23 = full_default_22 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_111, permute_114, permute_115, where_11, False, scale = 0.11180339887498948);  permute_111 = permute_114 = permute_115 = where_11 = None
        getitem_145: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_116: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_185: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_116, [32, 128, -1]);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_186: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [4096, 2560]);  view_185 = None
        permute_117: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg188_1, view_186, permute_117);  arg188_1 = view_186 = permute_117 = None
        view_187: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [32, 128, 2560]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_85: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_82, view_187);  add_82 = view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_278: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_278, [2], correction = 0, keepdim = True)
        getitem_154: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_23[0]
        getitem_155: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_23: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_278, getitem_155);  convert_element_type_278 = getitem_155 = None
        add_86: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_80: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_81: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, arg189_1);  mul_80 = arg189_1 = None
        add_87: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, arg190_1);  mul_81 = arg190_1 = None
        convert_element_type_279: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_188: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_279, [4096, 2560]);  convert_element_type_279 = None
        permute_118: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, view_188, permute_118);  arg192_1 = view_188 = permute_118 = None
        view_189: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 128, 10240]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_283: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_189, torch.float32);  view_189 = None
        mul_82: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_283, 0.5)
        mul_83: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_283, 0.7071067811865476);  convert_element_type_283 = None
        erf_11: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_88: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, add_88);  mul_82 = add_88 = None
        convert_element_type_284: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_190: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_284, [4096, 10240]);  convert_element_type_284 = None
        permute_119: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg194_1, view_190, permute_119);  arg194_1 = view_190 = permute_119 = None
        view_191: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [32, 128, 2560]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_89: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_85, view_191);  add_85 = view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_288: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_288, [2], correction = 0, keepdim = True)
        getitem_156: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_24[0]
        getitem_157: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        sub_24: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_288, getitem_157);  convert_element_type_288 = getitem_157 = None
        add_90: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_85: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_86: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, arg195_1);  mul_85 = arg195_1 = None
        add_91: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, arg196_1);  mul_86 = arg196_1 = None
        convert_element_type_289: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_192: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [4096, 2560])
        permute_120: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_72: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg198_1, view_192, permute_120);  arg198_1 = view_192 = permute_120 = None
        view_193: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [32, 128, 2560]);  addmm_72 = None
        view_194: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_193, [32, 128, -1, 80]);  view_193 = None
        permute_121: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_195: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [4096, 2560])
        permute_122: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_73: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg200_1, view_195, permute_122);  arg200_1 = view_195 = permute_122 = None
        view_196: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [32, 128, 2560]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_199: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_196, [32, 128, -1, 80]);  view_196 = None
        permute_124: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_197: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [4096, 2560]);  convert_element_type_289 = None
        permute_123: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_74: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg202_1, view_197, permute_123);  arg202_1 = view_197 = permute_123 = None
        view_198: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [32, 128, 2560]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_200: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_198, [32, 128, -1, 80]);  view_198 = None
        permute_125: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_25: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_25, full_default_24);  full_default_25 = full_default_24 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_121, permute_124, permute_125, where_12, False, scale = 0.11180339887498948);  permute_121 = permute_124 = permute_125 = where_12 = None
        getitem_158: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_126: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_201: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_126, [32, 128, -1]);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_202: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_201, [4096, 2560]);  view_201 = None
        permute_127: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_75: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg204_1, view_202, permute_127);  arg204_1 = view_202 = permute_127 = None
        view_203: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [32, 128, 2560]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_92: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_89, view_203);  add_89 = view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_302: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_92, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_302, [2], correction = 0, keepdim = True)
        getitem_167: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_25[0]
        getitem_168: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        sub_25: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_302, getitem_168);  convert_element_type_302 = getitem_168 = None
        add_93: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_167, 1e-05);  getitem_167 = None
        rsqrt_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_87: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_88: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, arg205_1);  mul_87 = arg205_1 = None
        add_94: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, arg206_1);  mul_88 = arg206_1 = None
        convert_element_type_303: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_204: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_303, [4096, 2560]);  convert_element_type_303 = None
        permute_128: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_76: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg208_1, view_204, permute_128);  arg208_1 = view_204 = permute_128 = None
        view_205: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [32, 128, 10240]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_307: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        mul_89: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_307, 0.5)
        mul_90: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_307, 0.7071067811865476);  convert_element_type_307 = None
        erf_12: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_95: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_91: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, add_95);  mul_89 = add_95 = None
        convert_element_type_308: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_91, torch.bfloat16);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_206: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_308, [4096, 10240]);  convert_element_type_308 = None
        permute_129: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_77: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg210_1, view_206, permute_129);  arg210_1 = view_206 = permute_129 = None
        view_207: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [32, 128, 2560]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_96: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_92, view_207);  add_92 = view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_312: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_312, [2], correction = 0, keepdim = True)
        getitem_169: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_26[0]
        getitem_170: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        sub_26: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_312, getitem_170);  convert_element_type_312 = getitem_170 = None
        add_97: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_169, 1e-05);  getitem_169 = None
        rsqrt_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_92: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
        mul_93: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, arg211_1);  mul_92 = arg211_1 = None
        add_98: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, arg212_1);  mul_93 = arg212_1 = None
        convert_element_type_313: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16);  add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_208: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_313, [4096, 2560])
        permute_130: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_78: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg214_1, view_208, permute_130);  arg214_1 = view_208 = permute_130 = None
        view_209: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [32, 128, 2560]);  addmm_78 = None
        view_210: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_209, [32, 128, -1, 80]);  view_209 = None
        permute_131: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_211: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_313, [4096, 2560])
        permute_132: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_79: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg216_1, view_211, permute_132);  arg216_1 = view_211 = permute_132 = None
        view_212: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [32, 128, 2560]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_215: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_212, [32, 128, -1, 80]);  view_212 = None
        permute_134: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_213: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_313, [4096, 2560]);  convert_element_type_313 = None
        permute_133: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_80: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg218_1, view_213, permute_133);  arg218_1 = view_213 = permute_133 = None
        view_214: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [32, 128, 2560]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_216: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_214, [32, 128, -1, 80]);  view_214 = None
        permute_135: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_27: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_27, full_default_26);  full_default_27 = full_default_26 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_131, permute_134, permute_135, where_13, False, scale = 0.11180339887498948);  permute_131 = permute_134 = permute_135 = where_13 = None
        getitem_171: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_136: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_217: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_136, [32, 128, -1]);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_218: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_217, [4096, 2560]);  view_217 = None
        permute_137: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_81: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg220_1, view_218, permute_137);  arg220_1 = view_218 = permute_137 = None
        view_219: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [32, 128, 2560]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_99: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_96, view_219);  add_96 = view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_326: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_326, [2], correction = 0, keepdim = True)
        getitem_180: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_27[0]
        getitem_181: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        sub_27: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_326, getitem_181);  convert_element_type_326 = getitem_181 = None
        add_100: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_94: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_95: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, arg221_1);  mul_94 = arg221_1 = None
        add_101: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, arg222_1);  mul_95 = arg222_1 = None
        convert_element_type_327: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_220: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_327, [4096, 2560]);  convert_element_type_327 = None
        permute_138: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_82: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg224_1, view_220, permute_138);  arg224_1 = view_220 = permute_138 = None
        view_221: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [32, 128, 10240]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_331: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_221, torch.float32);  view_221 = None
        mul_96: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_331, 0.5)
        mul_97: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_331, 0.7071067811865476);  convert_element_type_331 = None
        erf_13: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_102: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_98: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, add_102);  mul_96 = add_102 = None
        convert_element_type_332: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_222: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_332, [4096, 10240]);  convert_element_type_332 = None
        permute_139: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_83: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg226_1, view_222, permute_139);  arg226_1 = view_222 = permute_139 = None
        view_223: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [32, 128, 2560]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_103: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_99, view_223);  add_99 = view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_336: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_336, [2], correction = 0, keepdim = True)
        getitem_182: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_28[0]
        getitem_183: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        sub_28: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_336, getitem_183);  convert_element_type_336 = getitem_183 = None
        add_104: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_99: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_100: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, arg227_1);  mul_99 = arg227_1 = None
        add_105: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_100, arg228_1);  mul_100 = arg228_1 = None
        convert_element_type_337: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_224: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [4096, 2560])
        permute_140: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_84: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg230_1, view_224, permute_140);  arg230_1 = view_224 = permute_140 = None
        view_225: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [32, 128, 2560]);  addmm_84 = None
        view_226: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [32, 128, -1, 80]);  view_225 = None
        permute_141: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_227: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [4096, 2560])
        permute_142: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_85: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg232_1, view_227, permute_142);  arg232_1 = view_227 = permute_142 = None
        view_228: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_85, [32, 128, 2560]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_231: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_228, [32, 128, -1, 80]);  view_228 = None
        permute_144: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_229: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [4096, 2560]);  convert_element_type_337 = None
        permute_143: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_86: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg234_1, view_229, permute_143);  arg234_1 = view_229 = permute_143 = None
        view_230: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [32, 128, 2560]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_232: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_230, [32, 128, -1, 80]);  view_230 = None
        permute_145: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_29: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_29, full_default_28);  full_default_29 = full_default_28 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_141, permute_144, permute_145, where_14, False, scale = 0.11180339887498948);  permute_141 = permute_144 = permute_145 = where_14 = None
        getitem_184: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_146: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_184, [0, 2, 1, 3]);  getitem_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_233: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_146, [32, 128, -1]);  permute_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_234: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_233, [4096, 2560]);  view_233 = None
        permute_147: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_87: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg236_1, view_234, permute_147);  arg236_1 = view_234 = permute_147 = None
        view_235: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_87, [32, 128, 2560]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_106: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_103, view_235);  add_103 = view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_350: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_350, [2], correction = 0, keepdim = True)
        getitem_193: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_29[0]
        getitem_194: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        sub_29: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_350, getitem_194);  convert_element_type_350 = getitem_194 = None
        add_107: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_193, 1e-05);  getitem_193 = None
        rsqrt_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_101: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_102: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, arg237_1);  mul_101 = arg237_1 = None
        add_108: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, arg238_1);  mul_102 = arg238_1 = None
        convert_element_type_351: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16);  add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_236: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_351, [4096, 2560]);  convert_element_type_351 = None
        permute_148: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_88: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg240_1, view_236, permute_148);  arg240_1 = view_236 = permute_148 = None
        view_237: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [32, 128, 10240]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_355: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None
        mul_103: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.5)
        mul_104: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.7071067811865476);  convert_element_type_355 = None
        erf_14: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_109: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_105: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, add_109);  mul_103 = add_109 = None
        convert_element_type_356: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_238: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_356, [4096, 10240]);  convert_element_type_356 = None
        permute_149: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_89: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg242_1, view_238, permute_149);  arg242_1 = view_238 = permute_149 = None
        view_239: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_89, [32, 128, 2560]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_110: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_106, view_239);  add_106 = view_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_360: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_110, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_360, [2], correction = 0, keepdim = True)
        getitem_195: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_30[0]
        getitem_196: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        sub_30: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_360, getitem_196);  convert_element_type_360 = getitem_196 = None
        add_111: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_195, 1e-05);  getitem_195 = None
        rsqrt_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_106: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_107: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, arg243_1);  mul_106 = arg243_1 = None
        add_112: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_107, arg244_1);  mul_107 = arg244_1 = None
        convert_element_type_361: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_240: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [4096, 2560])
        permute_150: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_90: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg246_1, view_240, permute_150);  arg246_1 = view_240 = permute_150 = None
        view_241: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [32, 128, 2560]);  addmm_90 = None
        view_242: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_241, [32, 128, -1, 80]);  view_241 = None
        permute_151: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_243: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [4096, 2560])
        permute_152: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_91: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg248_1, view_243, permute_152);  arg248_1 = view_243 = permute_152 = None
        view_244: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_91, [32, 128, 2560]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_247: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_244, [32, 128, -1, 80]);  view_244 = None
        permute_154: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_245: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [4096, 2560]);  convert_element_type_361 = None
        permute_153: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_92: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg250_1, view_245, permute_153);  arg250_1 = view_245 = permute_153 = None
        view_246: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_92, [32, 128, 2560]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_248: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_246, [32, 128, -1, 80]);  view_246 = None
        permute_155: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_31: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_31, full_default_30);  full_default_31 = full_default_30 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_151, permute_154, permute_155, where_15, False, scale = 0.11180339887498948);  permute_151 = permute_154 = permute_155 = where_15 = None
        getitem_197: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_156: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_249: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_156, [32, 128, -1]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_250: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_249, [4096, 2560]);  view_249 = None
        permute_157: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_93: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg252_1, view_250, permute_157);  arg252_1 = view_250 = permute_157 = None
        view_251: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_93, [32, 128, 2560]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_113: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, view_251);  add_110 = view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_374: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_374, [2], correction = 0, keepdim = True)
        getitem_206: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_31[0]
        getitem_207: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        sub_31: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_374, getitem_207);  convert_element_type_374 = getitem_207 = None
        add_114: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_108: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_109: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, arg253_1);  mul_108 = arg253_1 = None
        add_115: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, arg254_1);  mul_109 = arg254_1 = None
        convert_element_type_375: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_252: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_375, [4096, 2560]);  convert_element_type_375 = None
        permute_158: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_94: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg256_1, view_252, permute_158);  arg256_1 = view_252 = permute_158 = None
        view_253: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [32, 128, 10240]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_379: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        mul_110: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_379, 0.5)
        mul_111: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_379, 0.7071067811865476);  convert_element_type_379 = None
        erf_15: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_116: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_112: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, add_116);  mul_110 = add_116 = None
        convert_element_type_380: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_254: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_380, [4096, 10240]);  convert_element_type_380 = None
        permute_159: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_95: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg258_1, view_254, permute_159);  arg258_1 = view_254 = permute_159 = None
        view_255: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_95, [32, 128, 2560]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_117: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_113, view_255);  add_113 = view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_384: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_384, [2], correction = 0, keepdim = True)
        getitem_208: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_32[0]
        getitem_209: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        sub_32: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_384, getitem_209);  convert_element_type_384 = getitem_209 = None
        add_118: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_113: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_114: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, arg259_1);  mul_113 = arg259_1 = None
        add_119: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, arg260_1);  mul_114 = arg260_1 = None
        convert_element_type_385: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_256: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_385, [4096, 2560])
        permute_160: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_96: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg262_1, view_256, permute_160);  arg262_1 = view_256 = permute_160 = None
        view_257: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_96, [32, 128, 2560]);  addmm_96 = None
        view_258: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [32, 128, -1, 80]);  view_257 = None
        permute_161: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_259: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_385, [4096, 2560])
        permute_162: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_97: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg264_1, view_259, permute_162);  arg264_1 = view_259 = permute_162 = None
        view_260: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_97, [32, 128, 2560]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_263: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_260, [32, 128, -1, 80]);  view_260 = None
        permute_164: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_261: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_385, [4096, 2560]);  convert_element_type_385 = None
        permute_163: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_98: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg266_1, view_261, permute_163);  arg266_1 = view_261 = permute_163 = None
        view_262: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_98, [32, 128, 2560]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_264: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_262, [32, 128, -1, 80]);  view_262 = None
        permute_165: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_33: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_33, full_default_32);  full_default_33 = full_default_32 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_161, permute_164, permute_165, where_16, False, scale = 0.11180339887498948);  permute_161 = permute_164 = permute_165 = where_16 = None
        getitem_210: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_16[0];  _scaled_dot_product_cudnn_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_166: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_210, [0, 2, 1, 3]);  getitem_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_265: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_166, [32, 128, -1]);  permute_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_266: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_265, [4096, 2560]);  view_265 = None
        permute_167: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_99: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg268_1, view_266, permute_167);  arg268_1 = view_266 = permute_167 = None
        view_267: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_99, [32, 128, 2560]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_120: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, view_267);  add_117 = view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_398: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_398, [2], correction = 0, keepdim = True)
        getitem_219: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_33[0]
        getitem_220: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        sub_33: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_398, getitem_220);  convert_element_type_398 = getitem_220 = None
        add_121: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_219, 1e-05);  getitem_219 = None
        rsqrt_33: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        mul_115: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_116: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, arg269_1);  mul_115 = arg269_1 = None
        add_122: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, arg270_1);  mul_116 = arg270_1 = None
        convert_element_type_399: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16);  add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_268: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_399, [4096, 2560]);  convert_element_type_399 = None
        permute_168: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_100: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg272_1, view_268, permute_168);  arg272_1 = view_268 = permute_168 = None
        view_269: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_100, [32, 128, 10240]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_403: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None
        mul_117: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, 0.5)
        mul_118: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, 0.7071067811865476);  convert_element_type_403 = None
        erf_16: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_123: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_119: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, add_123);  mul_117 = add_123 = None
        convert_element_type_404: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_270: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_404, [4096, 10240]);  convert_element_type_404 = None
        permute_169: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_101: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg274_1, view_270, permute_169);  arg274_1 = view_270 = permute_169 = None
        view_271: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_101, [32, 128, 2560]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_124: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_120, view_271);  add_120 = view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_408: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_408, [2], correction = 0, keepdim = True)
        getitem_221: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_34[0]
        getitem_222: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        sub_34: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_408, getitem_222);  convert_element_type_408 = getitem_222 = None
        add_125: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_221, 1e-05);  getitem_221 = None
        rsqrt_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_120: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_121: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, arg275_1);  mul_120 = arg275_1 = None
        add_126: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, arg276_1);  mul_121 = arg276_1 = None
        convert_element_type_409: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_126, torch.bfloat16);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_272: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_409, [4096, 2560])
        permute_170: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_102: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg278_1, view_272, permute_170);  arg278_1 = view_272 = permute_170 = None
        view_273: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_102, [32, 128, 2560]);  addmm_102 = None
        view_274: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_273, [32, 128, -1, 80]);  view_273 = None
        permute_171: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_275: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_409, [4096, 2560])
        permute_172: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_103: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg280_1, view_275, permute_172);  arg280_1 = view_275 = permute_172 = None
        view_276: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_103, [32, 128, 2560]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_279: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_276, [32, 128, -1, 80]);  view_276 = None
        permute_174: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_277: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_409, [4096, 2560]);  convert_element_type_409 = None
        permute_173: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_104: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg282_1, view_277, permute_173);  arg282_1 = view_277 = permute_173 = None
        view_278: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_104, [32, 128, 2560]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_280: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_278, [32, 128, -1, 80]);  view_278 = None
        permute_175: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_35: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_35, full_default_34);  full_default_35 = full_default_34 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_171, permute_174, permute_175, where_17, False, scale = 0.11180339887498948);  permute_171 = permute_174 = permute_175 = where_17 = None
        getitem_223: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_17[0];  _scaled_dot_product_cudnn_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_176: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_223, [0, 2, 1, 3]);  getitem_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_281: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_176, [32, 128, -1]);  permute_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_282: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_281, [4096, 2560]);  view_281 = None
        permute_177: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_105: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg284_1, view_282, permute_177);  arg284_1 = view_282 = permute_177 = None
        view_283: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_105, [32, 128, 2560]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_127: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_124, view_283);  add_124 = view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_422: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_422, [2], correction = 0, keepdim = True)
        getitem_232: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_35[0]
        getitem_233: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        sub_35: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_422, getitem_233);  convert_element_type_422 = getitem_233 = None
        add_128: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_232, 1e-05);  getitem_232 = None
        rsqrt_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_122: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_123: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, arg285_1);  mul_122 = arg285_1 = None
        add_129: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, arg286_1);  mul_123 = arg286_1 = None
        convert_element_type_423: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_284: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_423, [4096, 2560]);  convert_element_type_423 = None
        permute_178: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_106: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg288_1, view_284, permute_178);  arg288_1 = view_284 = permute_178 = None
        view_285: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_106, [32, 128, 10240]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_427: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        mul_124: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_427, 0.5)
        mul_125: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_427, 0.7071067811865476);  convert_element_type_427 = None
        erf_17: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_130: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_126: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, add_130);  mul_124 = add_130 = None
        convert_element_type_428: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_286: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_428, [4096, 10240]);  convert_element_type_428 = None
        permute_179: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_107: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg290_1, view_286, permute_179);  arg290_1 = view_286 = permute_179 = None
        view_287: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_107, [32, 128, 2560]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_131: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_127, view_287);  add_127 = view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_432: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_432, [2], correction = 0, keepdim = True)
        getitem_234: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_36[0]
        getitem_235: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        sub_36: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_432, getitem_235);  convert_element_type_432 = getitem_235 = None
        add_132: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_234, 1e-05);  getitem_234 = None
        rsqrt_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_127: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_128: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, arg291_1);  mul_127 = arg291_1 = None
        add_133: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_128, arg292_1);  mul_128 = arg292_1 = None
        convert_element_type_433: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_288: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_433, [4096, 2560])
        permute_180: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_108: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg294_1, view_288, permute_180);  arg294_1 = view_288 = permute_180 = None
        view_289: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_108, [32, 128, 2560]);  addmm_108 = None
        view_290: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_289, [32, 128, -1, 80]);  view_289 = None
        permute_181: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_291: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_433, [4096, 2560])
        permute_182: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_109: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg296_1, view_291, permute_182);  arg296_1 = view_291 = permute_182 = None
        view_292: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_109, [32, 128, 2560]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_295: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_292, [32, 128, -1, 80]);  view_292 = None
        permute_184: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_293: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_433, [4096, 2560]);  convert_element_type_433 = None
        permute_183: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_110: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg298_1, view_293, permute_183);  arg298_1 = view_293 = permute_183 = None
        view_294: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_110, [32, 128, 2560]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_296: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_294, [32, 128, -1, 80]);  view_294 = None
        permute_185: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_37: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_37, full_default_36);  full_default_37 = full_default_36 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_181, permute_184, permute_185, where_18, False, scale = 0.11180339887498948);  permute_181 = permute_184 = permute_185 = where_18 = None
        getitem_236: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_18[0];  _scaled_dot_product_cudnn_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_186: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_236, [0, 2, 1, 3]);  getitem_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_297: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_186, [32, 128, -1]);  permute_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_298: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_297, [4096, 2560]);  view_297 = None
        permute_187: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_111: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg300_1, view_298, permute_187);  arg300_1 = view_298 = permute_187 = None
        view_299: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_111, [32, 128, 2560]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_134: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_131, view_299);  add_131 = view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_446: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_446, [2], correction = 0, keepdim = True)
        getitem_245: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_37[0]
        getitem_246: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        sub_37: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_446, getitem_246);  convert_element_type_446 = getitem_246 = None
        add_135: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_245, 1e-05);  getitem_245 = None
        rsqrt_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_129: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_130: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, arg301_1);  mul_129 = arg301_1 = None
        add_136: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, arg302_1);  mul_130 = arg302_1 = None
        convert_element_type_447: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_300: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_447, [4096, 2560]);  convert_element_type_447 = None
        permute_188: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_112: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg304_1, view_300, permute_188);  arg304_1 = view_300 = permute_188 = None
        view_301: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_112, [32, 128, 10240]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_451: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_301, torch.float32);  view_301 = None
        mul_131: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_451, 0.5)
        mul_132: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_451, 0.7071067811865476);  convert_element_type_451 = None
        erf_18: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_137: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_133: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, add_137);  mul_131 = add_137 = None
        convert_element_type_452: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_302: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_452, [4096, 10240]);  convert_element_type_452 = None
        permute_189: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_113: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg306_1, view_302, permute_189);  arg306_1 = view_302 = permute_189 = None
        view_303: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_113, [32, 128, 2560]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_138: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, view_303);  add_134 = view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_456: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_456, [2], correction = 0, keepdim = True)
        getitem_247: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_38[0]
        getitem_248: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        sub_38: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_456, getitem_248);  convert_element_type_456 = getitem_248 = None
        add_139: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_247, 1e-05);  getitem_247 = None
        rsqrt_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_134: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_135: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, arg307_1);  mul_134 = arg307_1 = None
        add_140: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_135, arg308_1);  mul_135 = arg308_1 = None
        convert_element_type_457: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_140, torch.bfloat16);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_304: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [4096, 2560])
        permute_190: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_114: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg310_1, view_304, permute_190);  arg310_1 = view_304 = permute_190 = None
        view_305: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_114, [32, 128, 2560]);  addmm_114 = None
        view_306: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_305, [32, 128, -1, 80]);  view_305 = None
        permute_191: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_307: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [4096, 2560])
        permute_192: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_115: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg312_1, view_307, permute_192);  arg312_1 = view_307 = permute_192 = None
        view_308: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_115, [32, 128, 2560]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_311: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_308, [32, 128, -1, 80]);  view_308 = None
        permute_194: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_309: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [4096, 2560]);  convert_element_type_457 = None
        permute_193: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_116: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg314_1, view_309, permute_193);  arg314_1 = view_309 = permute_193 = None
        view_310: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_116, [32, 128, 2560]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_312: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_310, [32, 128, -1, 80]);  view_310 = None
        permute_195: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_39: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_39, full_default_38);  full_default_39 = full_default_38 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_191, permute_194, permute_195, where_19, False, scale = 0.11180339887498948);  permute_191 = permute_194 = permute_195 = where_19 = None
        getitem_249: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_19[0];  _scaled_dot_product_cudnn_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_196: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_249, [0, 2, 1, 3]);  getitem_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_313: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_196, [32, 128, -1]);  permute_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_314: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_313, [4096, 2560]);  view_313 = None
        permute_197: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_117: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg316_1, view_314, permute_197);  arg316_1 = view_314 = permute_197 = None
        view_315: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_117, [32, 128, 2560]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_141: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_138, view_315);  add_138 = view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_470: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_470, [2], correction = 0, keepdim = True)
        getitem_258: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_39[0]
        getitem_259: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        sub_39: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_470, getitem_259);  convert_element_type_470 = getitem_259 = None
        add_142: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_258, 1e-05);  getitem_258 = None
        rsqrt_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_136: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_137: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, arg317_1);  mul_136 = arg317_1 = None
        add_143: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, arg318_1);  mul_137 = arg318_1 = None
        convert_element_type_471: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_316: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_471, [4096, 2560]);  convert_element_type_471 = None
        permute_198: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_118: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg320_1, view_316, permute_198);  arg320_1 = view_316 = permute_198 = None
        view_317: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_118, [32, 128, 10240]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_475: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None
        mul_138: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, 0.5)
        mul_139: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, 0.7071067811865476);  convert_element_type_475 = None
        erf_19: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
        add_144: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_140: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, add_144);  mul_138 = add_144 = None
        convert_element_type_476: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_140, torch.bfloat16);  mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_318: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_476, [4096, 10240]);  convert_element_type_476 = None
        permute_199: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_119: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg322_1, view_318, permute_199);  arg322_1 = view_318 = permute_199 = None
        view_319: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_119, [32, 128, 2560]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_145: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, view_319);  add_141 = view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_480: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_480, [2], correction = 0, keepdim = True)
        getitem_260: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_40[0]
        getitem_261: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        sub_40: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_480, getitem_261);  convert_element_type_480 = getitem_261 = None
        add_146: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_260, 1e-05);  getitem_260 = None
        rsqrt_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_141: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_142: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, arg323_1);  mul_141 = arg323_1 = None
        add_147: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, arg324_1);  mul_142 = arg324_1 = None
        convert_element_type_481: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.bfloat16);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_320: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [4096, 2560])
        permute_200: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_120: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg326_1, view_320, permute_200);  arg326_1 = view_320 = permute_200 = None
        view_321: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_120, [32, 128, 2560]);  addmm_120 = None
        view_322: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_321, [32, 128, -1, 80]);  view_321 = None
        permute_201: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_323: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [4096, 2560])
        permute_202: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_121: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg328_1, view_323, permute_202);  arg328_1 = view_323 = permute_202 = None
        view_324: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_121, [32, 128, 2560]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_327: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_324, [32, 128, -1, 80]);  view_324 = None
        permute_204: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_325: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [4096, 2560]);  convert_element_type_481 = None
        permute_203: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_122: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg330_1, view_325, permute_203);  arg330_1 = view_325 = permute_203 = None
        view_326: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_122, [32, 128, 2560]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_328: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_326, [32, 128, -1, 80]);  view_326 = None
        permute_205: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_41: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_41, full_default_40);  full_default_41 = full_default_40 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_201, permute_204, permute_205, where_20, False, scale = 0.11180339887498948);  permute_201 = permute_204 = permute_205 = where_20 = None
        getitem_262: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_20[0];  _scaled_dot_product_cudnn_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_206: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_262, [0, 2, 1, 3]);  getitem_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_329: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_206, [32, 128, -1]);  permute_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_330: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [4096, 2560]);  view_329 = None
        permute_207: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_123: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg332_1, view_330, permute_207);  arg332_1 = view_330 = permute_207 = None
        view_331: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_123, [32, 128, 2560]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_148: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, view_331);  add_145 = view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_494: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_148, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_494, [2], correction = 0, keepdim = True)
        getitem_271: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_41[0]
        getitem_272: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        sub_41: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_494, getitem_272);  convert_element_type_494 = getitem_272 = None
        add_149: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_271, 1e-05);  getitem_271 = None
        rsqrt_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_143: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_144: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, arg333_1);  mul_143 = arg333_1 = None
        add_150: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, arg334_1);  mul_144 = arg334_1 = None
        convert_element_type_495: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_150, torch.bfloat16);  add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_332: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_495, [4096, 2560]);  convert_element_type_495 = None
        permute_208: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_124: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg336_1, view_332, permute_208);  arg336_1 = view_332 = permute_208 = None
        view_333: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_124, [32, 128, 10240]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_499: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_333, torch.float32);  view_333 = None
        mul_145: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_499, 0.5)
        mul_146: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_499, 0.7071067811865476);  convert_element_type_499 = None
        erf_20: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
        add_151: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_147: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, add_151);  mul_145 = add_151 = None
        convert_element_type_500: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_334: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_500, [4096, 10240]);  convert_element_type_500 = None
        permute_209: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_125: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg338_1, view_334, permute_209);  arg338_1 = view_334 = permute_209 = None
        view_335: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_125, [32, 128, 2560]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_152: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_148, view_335);  add_148 = view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_504: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_504, [2], correction = 0, keepdim = True)
        getitem_273: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_42[0]
        getitem_274: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        sub_42: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_504, getitem_274);  convert_element_type_504 = getitem_274 = None
        add_153: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_273, 1e-05);  getitem_273 = None
        rsqrt_42: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_148: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_149: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, arg339_1);  mul_148 = arg339_1 = None
        add_154: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, arg340_1);  mul_149 = arg340_1 = None
        convert_element_type_505: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_336: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [4096, 2560])
        permute_210: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_126: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg342_1, view_336, permute_210);  arg342_1 = view_336 = permute_210 = None
        view_337: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_126, [32, 128, 2560]);  addmm_126 = None
        view_338: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_337, [32, 128, -1, 80]);  view_337 = None
        permute_211: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_339: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [4096, 2560])
        permute_212: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_127: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg344_1, view_339, permute_212);  arg344_1 = view_339 = permute_212 = None
        view_340: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_127, [32, 128, 2560]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_343: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_340, [32, 128, -1, 80]);  view_340 = None
        permute_214: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_341: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [4096, 2560]);  convert_element_type_505 = None
        permute_213: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_128: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg346_1, view_341, permute_213);  arg346_1 = view_341 = permute_213 = None
        view_342: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_128, [32, 128, 2560]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_344: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_342, [32, 128, -1, 80]);  view_342 = None
        permute_215: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_43: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_43, full_default_42);  full_default_43 = full_default_42 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_211, permute_214, permute_215, where_21, False, scale = 0.11180339887498948);  permute_211 = permute_214 = permute_215 = where_21 = None
        getitem_275: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_21[0];  _scaled_dot_product_cudnn_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_216: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_275, [0, 2, 1, 3]);  getitem_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_345: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_216, [32, 128, -1]);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_346: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [4096, 2560]);  view_345 = None
        permute_217: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_129: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg348_1, view_346, permute_217);  arg348_1 = view_346 = permute_217 = None
        view_347: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_129, [32, 128, 2560]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_155: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_152, view_347);  add_152 = view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_518: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_155, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_518, [2], correction = 0, keepdim = True)
        getitem_284: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_43[0]
        getitem_285: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        sub_43: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_518, getitem_285);  convert_element_type_518 = getitem_285 = None
        add_156: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_284, 1e-05);  getitem_284 = None
        rsqrt_43: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        mul_150: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_151: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, arg349_1);  mul_150 = arg349_1 = None
        add_157: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, arg350_1);  mul_151 = arg350_1 = None
        convert_element_type_519: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_348: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_519, [4096, 2560]);  convert_element_type_519 = None
        permute_218: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_130: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg352_1, view_348, permute_218);  arg352_1 = view_348 = permute_218 = None
        view_349: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_130, [32, 128, 10240]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_523: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        mul_152: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_523, 0.5)
        mul_153: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_523, 0.7071067811865476);  convert_element_type_523 = None
        erf_21: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_158: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_154: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, add_158);  mul_152 = add_158 = None
        convert_element_type_524: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_350: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_524, [4096, 10240]);  convert_element_type_524 = None
        permute_219: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_131: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg354_1, view_350, permute_219);  arg354_1 = view_350 = permute_219 = None
        view_351: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_131, [32, 128, 2560]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_159: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_155, view_351);  add_155 = view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_528: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_528, [2], correction = 0, keepdim = True)
        getitem_286: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_44[0]
        getitem_287: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        sub_44: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_528, getitem_287);  convert_element_type_528 = getitem_287 = None
        add_160: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_286, 1e-05);  getitem_286 = None
        rsqrt_44: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_155: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_156: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, arg355_1);  mul_155 = arg355_1 = None
        add_161: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_156, arg356_1);  mul_156 = arg356_1 = None
        convert_element_type_529: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.bfloat16);  add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_352: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_529, [4096, 2560])
        permute_220: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_132: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg358_1, view_352, permute_220);  arg358_1 = view_352 = permute_220 = None
        view_353: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_132, [32, 128, 2560]);  addmm_132 = None
        view_354: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_353, [32, 128, -1, 80]);  view_353 = None
        permute_221: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_355: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_529, [4096, 2560])
        permute_222: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_133: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg360_1, view_355, permute_222);  arg360_1 = view_355 = permute_222 = None
        view_356: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_133, [32, 128, 2560]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_359: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_356, [32, 128, -1, 80]);  view_356 = None
        permute_224: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_357: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_529, [4096, 2560]);  convert_element_type_529 = None
        permute_223: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_134: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg362_1, view_357, permute_223);  arg362_1 = view_357 = permute_223 = None
        view_358: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_134, [32, 128, 2560]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_360: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_358, [32, 128, -1, 80]);  view_358 = None
        permute_225: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_45: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_45, full_default_44);  full_default_45 = full_default_44 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_221, permute_224, permute_225, where_22, False, scale = 0.11180339887498948);  permute_221 = permute_224 = permute_225 = where_22 = None
        getitem_288: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_22[0];  _scaled_dot_product_cudnn_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_226: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_288, [0, 2, 1, 3]);  getitem_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_361: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_226, [32, 128, -1]);  permute_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_362: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_361, [4096, 2560]);  view_361 = None
        permute_227: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_135: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg364_1, view_362, permute_227);  arg364_1 = view_362 = permute_227 = None
        view_363: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_135, [32, 128, 2560]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_162: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_159, view_363);  add_159 = view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_542: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_162, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_542, [2], correction = 0, keepdim = True)
        getitem_297: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_45[0]
        getitem_298: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        sub_45: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_542, getitem_298);  convert_element_type_542 = getitem_298 = None
        add_163: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_297, 1e-05);  getitem_297 = None
        rsqrt_45: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_157: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_158: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, arg365_1);  mul_157 = arg365_1 = None
        add_164: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, arg366_1);  mul_158 = arg366_1 = None
        convert_element_type_543: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_364: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_543, [4096, 2560]);  convert_element_type_543 = None
        permute_228: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_136: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg368_1, view_364, permute_228);  arg368_1 = view_364 = permute_228 = None
        view_365: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_136, [32, 128, 10240]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_547: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_365, torch.float32);  view_365 = None
        mul_159: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_547, 0.5)
        mul_160: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_547, 0.7071067811865476);  convert_element_type_547 = None
        erf_22: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_165: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_161: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_159, add_165);  mul_159 = add_165 = None
        convert_element_type_548: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_366: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_548, [4096, 10240]);  convert_element_type_548 = None
        permute_229: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_137: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg370_1, view_366, permute_229);  arg370_1 = view_366 = permute_229 = None
        view_367: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_137, [32, 128, 2560]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_166: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_162, view_367);  add_162 = view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_552: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_166, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_552, [2], correction = 0, keepdim = True)
        getitem_299: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_46[0]
        getitem_300: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        sub_46: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_552, getitem_300);  convert_element_type_552 = getitem_300 = None
        add_167: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_299, 1e-05);  getitem_299 = None
        rsqrt_46: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
        mul_162: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_163: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, arg371_1);  mul_162 = arg371_1 = None
        add_168: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_163, arg372_1);  mul_163 = arg372_1 = None
        convert_element_type_553: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_168, torch.bfloat16);  add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_368: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_553, [4096, 2560])
        permute_230: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_138: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg374_1, view_368, permute_230);  arg374_1 = view_368 = permute_230 = None
        view_369: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_138, [32, 128, 2560]);  addmm_138 = None
        view_370: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_369, [32, 128, -1, 80]);  view_369 = None
        permute_231: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_371: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_553, [4096, 2560])
        permute_232: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        addmm_139: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg376_1, view_371, permute_232);  arg376_1 = view_371 = permute_232 = None
        view_372: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_139, [32, 128, 2560]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_375: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_372, [32, 128, -1, 80]);  view_372 = None
        permute_234: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_373: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_553, [4096, 2560]);  convert_element_type_553 = None
        permute_233: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_140: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg378_1, view_373, permute_233);  arg378_1 = view_373 = permute_233 = None
        view_374: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_140, [32, 128, 2560]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_376: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_374, [32, 128, -1, 80]);  view_374 = None
        permute_235: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_47: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_47, full_default_46);  expand = full_default_47 = full_default_46 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_231, permute_234, permute_235, where_23, False, scale = 0.11180339887498948);  permute_231 = permute_234 = permute_235 = where_23 = None
        getitem_301: "bf16[32, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_23[0];  _scaled_dot_product_cudnn_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_236: "bf16[32, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_301, [0, 2, 1, 3]);  getitem_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_377: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_236, [32, 128, -1]);  permute_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_378: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_377, [4096, 2560]);  view_377 = None
        permute_237: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_141: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg380_1, view_378, permute_237);  arg380_1 = view_378 = permute_237 = None
        view_379: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_141, [32, 128, 2560]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_169: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_166, view_379);  add_166 = view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_566: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_566, [2], correction = 0, keepdim = True)
        getitem_310: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_47[0]
        getitem_311: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        sub_47: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_566, getitem_311);  convert_element_type_566 = getitem_311 = None
        add_170: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_47: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_164: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_165: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, arg381_1);  mul_164 = arg381_1 = None
        add_171: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, arg382_1);  mul_165 = arg382_1 = None
        convert_element_type_567: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16);  add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_380: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_567, [4096, 2560]);  convert_element_type_567 = None
        permute_238: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_142: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg384_1, view_380, permute_238);  arg384_1 = view_380 = permute_238 = None
        view_381: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_142, [32, 128, 10240]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_571: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_381, torch.float32);  view_381 = None
        mul_166: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_571, 0.5)
        mul_167: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_571, 0.7071067811865476);  convert_element_type_571 = None
        erf_23: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_172: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_168: "f32[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, add_172);  mul_166 = add_172 = None
        convert_element_type_572: "bf16[32, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_382: "bf16[4096, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_572, [4096, 10240]);  convert_element_type_572 = None
        permute_239: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_143: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg386_1, view_382, permute_239);  arg386_1 = view_382 = permute_239 = None
        view_383: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_143, [32, 128, 2560]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_173: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_169, view_383);  add_169 = view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:638 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_576: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.float32);  add_173 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_576, [2], correction = 0, keepdim = True)
        getitem_312: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_48[0]
        getitem_313: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1004 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_387: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(arg389_1, [-1]);  arg389_1 = None
        ne_1: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_387, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:638 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_48: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_576, getitem_313);  convert_element_type_576 = getitem_313 = None
        add_174: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_312, 1e-05);  getitem_312 = None
        rsqrt_48: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_169: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_170: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, arg387_1);  mul_169 = arg387_1 = None
        add_175: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, arg388_1);  mul_170 = arg388_1 = None
        convert_element_type_577: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:998 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_384: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_577, [4096, 2560]);  convert_element_type_577 = None
        permute_240: "bf16[2560, 8008][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.mm.default(view_384, permute_240);  view_384 = permute_240 = None
        view_385: "bf16[32, 128, 8008][1025024, 8008, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 128, 8008]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1004 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_386: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.reshape.default(view_385, [-1, 8008])
        convert_element_type_580: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_386, torch.float32);  view_386 = None
        amax: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_580, [1], True)
        sub_49: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_580, amax);  convert_element_type_580 = amax = None
        exp: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.exp.default(sub_49)
        sum_1: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_50: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_49, log);  sub_49 = log = None
        convert_element_type_581: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_50, torch.bfloat16);  sub_50 = None
        ne: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_387, -100)
        full_default_48: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[4096][1]cuda:0" = torch.ops.aten.where.self(ne, view_387, full_default_48);  ne = full_default_48 = None
        unsqueeze_9: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather: "bf16[4096, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_581, 1, unsqueeze_9);  convert_element_type_581 = unsqueeze_9 = None
        squeeze: "bf16[4096][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[4096][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_49: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "bf16[4096][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_49);  ne_1 = neg = full_default_49 = None
        sum_3: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_25);  where_25 = None
        ne_2: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_387, -100);  view_387 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_582: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.bfloat16);  sum_2 = None
        div: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_582);  sum_3 = convert_element_type_582 = None
        return (div, view_385)

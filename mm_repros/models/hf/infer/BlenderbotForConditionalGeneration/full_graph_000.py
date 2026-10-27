class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 128][128, 1]cuda:0", arg1_1: "i64[16, 128][128, 1]cuda:0", arg2_1: "bf16[8008, 2560][2560, 1]cuda:0", arg3_1: "bf16[128, 2560][2560, 1]cuda:0", arg4_1: "bf16[2560][1]cuda:0", arg5_1: "bf16[2560][1]cuda:0", arg6_1: "bf16[2560, 2560][2560, 1]cuda:0", arg7_1: "bf16[2560][1]cuda:0", arg8_1: "bf16[2560, 2560][2560, 1]cuda:0", arg9_1: "bf16[2560][1]cuda:0", arg10_1: "bf16[2560, 2560][2560, 1]cuda:0", arg11_1: "bf16[2560][1]cuda:0", arg12_1: "bf16[2560, 2560][2560, 1]cuda:0", arg13_1: "bf16[2560][1]cuda:0", arg14_1: "bf16[2560][1]cuda:0", arg15_1: "bf16[2560][1]cuda:0", arg16_1: "bf16[10240, 2560][2560, 1]cuda:0", arg17_1: "bf16[10240][1]cuda:0", arg18_1: "bf16[2560, 10240][10240, 1]cuda:0", arg19_1: "bf16[2560][1]cuda:0", arg20_1: "bf16[2560][1]cuda:0", arg21_1: "bf16[2560][1]cuda:0", arg22_1: "bf16[2560, 2560][2560, 1]cuda:0", arg23_1: "bf16[2560][1]cuda:0", arg24_1: "bf16[2560, 2560][2560, 1]cuda:0", arg25_1: "bf16[2560][1]cuda:0", arg26_1: "bf16[2560, 2560][2560, 1]cuda:0", arg27_1: "bf16[2560][1]cuda:0", arg28_1: "bf16[2560, 2560][2560, 1]cuda:0", arg29_1: "bf16[2560][1]cuda:0", arg30_1: "bf16[2560][1]cuda:0", arg31_1: "bf16[2560][1]cuda:0", arg32_1: "bf16[10240, 2560][2560, 1]cuda:0", arg33_1: "bf16[10240][1]cuda:0", arg34_1: "bf16[2560, 10240][10240, 1]cuda:0", arg35_1: "bf16[2560][1]cuda:0", arg36_1: "bf16[2560][1]cuda:0", arg37_1: "bf16[2560][1]cuda:0", arg38_1: "bf16[128, 2560][2560, 1]cuda:0", arg39_1: "bf16[2560][1]cuda:0", arg40_1: "bf16[2560][1]cuda:0", arg41_1: "bf16[2560, 2560][2560, 1]cuda:0", arg42_1: "bf16[2560][1]cuda:0", arg43_1: "bf16[2560, 2560][2560, 1]cuda:0", arg44_1: "bf16[2560][1]cuda:0", arg45_1: "bf16[2560, 2560][2560, 1]cuda:0", arg46_1: "bf16[2560][1]cuda:0", arg47_1: "bf16[2560, 2560][2560, 1]cuda:0", arg48_1: "bf16[2560][1]cuda:0", arg49_1: "bf16[2560][1]cuda:0", arg50_1: "bf16[2560][1]cuda:0", arg51_1: "bf16[2560, 2560][2560, 1]cuda:0", arg52_1: "bf16[2560][1]cuda:0", arg53_1: "bf16[2560, 2560][2560, 1]cuda:0", arg54_1: "bf16[2560][1]cuda:0", arg55_1: "bf16[2560, 2560][2560, 1]cuda:0", arg56_1: "bf16[2560][1]cuda:0", arg57_1: "bf16[2560, 2560][2560, 1]cuda:0", arg58_1: "bf16[2560][1]cuda:0", arg59_1: "bf16[2560][1]cuda:0", arg60_1: "bf16[2560][1]cuda:0", arg61_1: "bf16[10240, 2560][2560, 1]cuda:0", arg62_1: "bf16[10240][1]cuda:0", arg63_1: "bf16[2560, 10240][10240, 1]cuda:0", arg64_1: "bf16[2560][1]cuda:0", arg65_1: "bf16[2560][1]cuda:0", arg66_1: "bf16[2560][1]cuda:0", arg67_1: "bf16[2560, 2560][2560, 1]cuda:0", arg68_1: "bf16[2560][1]cuda:0", arg69_1: "bf16[2560, 2560][2560, 1]cuda:0", arg70_1: "bf16[2560][1]cuda:0", arg71_1: "bf16[2560, 2560][2560, 1]cuda:0", arg72_1: "bf16[2560][1]cuda:0", arg73_1: "bf16[2560, 2560][2560, 1]cuda:0", arg74_1: "bf16[2560][1]cuda:0", arg75_1: "bf16[2560][1]cuda:0", arg76_1: "bf16[2560][1]cuda:0", arg77_1: "bf16[2560, 2560][2560, 1]cuda:0", arg78_1: "bf16[2560][1]cuda:0", arg79_1: "bf16[2560, 2560][2560, 1]cuda:0", arg80_1: "bf16[2560][1]cuda:0", arg81_1: "bf16[2560, 2560][2560, 1]cuda:0", arg82_1: "bf16[2560][1]cuda:0", arg83_1: "bf16[2560, 2560][2560, 1]cuda:0", arg84_1: "bf16[2560][1]cuda:0", arg85_1: "bf16[2560][1]cuda:0", arg86_1: "bf16[2560][1]cuda:0", arg87_1: "bf16[10240, 2560][2560, 1]cuda:0", arg88_1: "bf16[10240][1]cuda:0", arg89_1: "bf16[2560, 10240][10240, 1]cuda:0", arg90_1: "bf16[2560][1]cuda:0", arg91_1: "bf16[2560][1]cuda:0", arg92_1: "bf16[2560][1]cuda:0", arg93_1: "bf16[2560, 2560][2560, 1]cuda:0", arg94_1: "bf16[2560][1]cuda:0", arg95_1: "bf16[2560, 2560][2560, 1]cuda:0", arg96_1: "bf16[2560][1]cuda:0", arg97_1: "bf16[2560, 2560][2560, 1]cuda:0", arg98_1: "bf16[2560][1]cuda:0", arg99_1: "bf16[2560, 2560][2560, 1]cuda:0", arg100_1: "bf16[2560][1]cuda:0", arg101_1: "bf16[2560][1]cuda:0", arg102_1: "bf16[2560][1]cuda:0", arg103_1: "bf16[2560, 2560][2560, 1]cuda:0", arg104_1: "bf16[2560][1]cuda:0", arg105_1: "bf16[2560, 2560][2560, 1]cuda:0", arg106_1: "bf16[2560][1]cuda:0", arg107_1: "bf16[2560, 2560][2560, 1]cuda:0", arg108_1: "bf16[2560][1]cuda:0", arg109_1: "bf16[2560, 2560][2560, 1]cuda:0", arg110_1: "bf16[2560][1]cuda:0", arg111_1: "bf16[2560][1]cuda:0", arg112_1: "bf16[2560][1]cuda:0", arg113_1: "bf16[10240, 2560][2560, 1]cuda:0", arg114_1: "bf16[10240][1]cuda:0", arg115_1: "bf16[2560, 10240][10240, 1]cuda:0", arg116_1: "bf16[2560][1]cuda:0", arg117_1: "bf16[2560][1]cuda:0", arg118_1: "bf16[2560][1]cuda:0", arg119_1: "bf16[2560, 2560][2560, 1]cuda:0", arg120_1: "bf16[2560][1]cuda:0", arg121_1: "bf16[2560, 2560][2560, 1]cuda:0", arg122_1: "bf16[2560][1]cuda:0", arg123_1: "bf16[2560, 2560][2560, 1]cuda:0", arg124_1: "bf16[2560][1]cuda:0", arg125_1: "bf16[2560, 2560][2560, 1]cuda:0", arg126_1: "bf16[2560][1]cuda:0", arg127_1: "bf16[2560][1]cuda:0", arg128_1: "bf16[2560][1]cuda:0", arg129_1: "bf16[2560, 2560][2560, 1]cuda:0", arg130_1: "bf16[2560][1]cuda:0", arg131_1: "bf16[2560, 2560][2560, 1]cuda:0", arg132_1: "bf16[2560][1]cuda:0", arg133_1: "bf16[2560, 2560][2560, 1]cuda:0", arg134_1: "bf16[2560][1]cuda:0", arg135_1: "bf16[2560, 2560][2560, 1]cuda:0", arg136_1: "bf16[2560][1]cuda:0", arg137_1: "bf16[2560][1]cuda:0", arg138_1: "bf16[2560][1]cuda:0", arg139_1: "bf16[10240, 2560][2560, 1]cuda:0", arg140_1: "bf16[10240][1]cuda:0", arg141_1: "bf16[2560, 10240][10240, 1]cuda:0", arg142_1: "bf16[2560][1]cuda:0", arg143_1: "bf16[2560][1]cuda:0", arg144_1: "bf16[2560][1]cuda:0", arg145_1: "bf16[2560, 2560][2560, 1]cuda:0", arg146_1: "bf16[2560][1]cuda:0", arg147_1: "bf16[2560, 2560][2560, 1]cuda:0", arg148_1: "bf16[2560][1]cuda:0", arg149_1: "bf16[2560, 2560][2560, 1]cuda:0", arg150_1: "bf16[2560][1]cuda:0", arg151_1: "bf16[2560, 2560][2560, 1]cuda:0", arg152_1: "bf16[2560][1]cuda:0", arg153_1: "bf16[2560][1]cuda:0", arg154_1: "bf16[2560][1]cuda:0", arg155_1: "bf16[2560, 2560][2560, 1]cuda:0", arg156_1: "bf16[2560][1]cuda:0", arg157_1: "bf16[2560, 2560][2560, 1]cuda:0", arg158_1: "bf16[2560][1]cuda:0", arg159_1: "bf16[2560, 2560][2560, 1]cuda:0", arg160_1: "bf16[2560][1]cuda:0", arg161_1: "bf16[2560, 2560][2560, 1]cuda:0", arg162_1: "bf16[2560][1]cuda:0", arg163_1: "bf16[2560][1]cuda:0", arg164_1: "bf16[2560][1]cuda:0", arg165_1: "bf16[10240, 2560][2560, 1]cuda:0", arg166_1: "bf16[10240][1]cuda:0", arg167_1: "bf16[2560, 10240][10240, 1]cuda:0", arg168_1: "bf16[2560][1]cuda:0", arg169_1: "bf16[2560][1]cuda:0", arg170_1: "bf16[2560][1]cuda:0", arg171_1: "bf16[2560, 2560][2560, 1]cuda:0", arg172_1: "bf16[2560][1]cuda:0", arg173_1: "bf16[2560, 2560][2560, 1]cuda:0", arg174_1: "bf16[2560][1]cuda:0", arg175_1: "bf16[2560, 2560][2560, 1]cuda:0", arg176_1: "bf16[2560][1]cuda:0", arg177_1: "bf16[2560, 2560][2560, 1]cuda:0", arg178_1: "bf16[2560][1]cuda:0", arg179_1: "bf16[2560][1]cuda:0", arg180_1: "bf16[2560][1]cuda:0", arg181_1: "bf16[2560, 2560][2560, 1]cuda:0", arg182_1: "bf16[2560][1]cuda:0", arg183_1: "bf16[2560, 2560][2560, 1]cuda:0", arg184_1: "bf16[2560][1]cuda:0", arg185_1: "bf16[2560, 2560][2560, 1]cuda:0", arg186_1: "bf16[2560][1]cuda:0", arg187_1: "bf16[2560, 2560][2560, 1]cuda:0", arg188_1: "bf16[2560][1]cuda:0", arg189_1: "bf16[2560][1]cuda:0", arg190_1: "bf16[2560][1]cuda:0", arg191_1: "bf16[10240, 2560][2560, 1]cuda:0", arg192_1: "bf16[10240][1]cuda:0", arg193_1: "bf16[2560, 10240][10240, 1]cuda:0", arg194_1: "bf16[2560][1]cuda:0", arg195_1: "bf16[2560][1]cuda:0", arg196_1: "bf16[2560][1]cuda:0", arg197_1: "bf16[2560, 2560][2560, 1]cuda:0", arg198_1: "bf16[2560][1]cuda:0", arg199_1: "bf16[2560, 2560][2560, 1]cuda:0", arg200_1: "bf16[2560][1]cuda:0", arg201_1: "bf16[2560, 2560][2560, 1]cuda:0", arg202_1: "bf16[2560][1]cuda:0", arg203_1: "bf16[2560, 2560][2560, 1]cuda:0", arg204_1: "bf16[2560][1]cuda:0", arg205_1: "bf16[2560][1]cuda:0", arg206_1: "bf16[2560][1]cuda:0", arg207_1: "bf16[2560, 2560][2560, 1]cuda:0", arg208_1: "bf16[2560][1]cuda:0", arg209_1: "bf16[2560, 2560][2560, 1]cuda:0", arg210_1: "bf16[2560][1]cuda:0", arg211_1: "bf16[2560, 2560][2560, 1]cuda:0", arg212_1: "bf16[2560][1]cuda:0", arg213_1: "bf16[2560, 2560][2560, 1]cuda:0", arg214_1: "bf16[2560][1]cuda:0", arg215_1: "bf16[2560][1]cuda:0", arg216_1: "bf16[2560][1]cuda:0", arg217_1: "bf16[10240, 2560][2560, 1]cuda:0", arg218_1: "bf16[10240][1]cuda:0", arg219_1: "bf16[2560, 10240][10240, 1]cuda:0", arg220_1: "bf16[2560][1]cuda:0", arg221_1: "bf16[2560][1]cuda:0", arg222_1: "bf16[2560][1]cuda:0", arg223_1: "bf16[2560, 2560][2560, 1]cuda:0", arg224_1: "bf16[2560][1]cuda:0", arg225_1: "bf16[2560, 2560][2560, 1]cuda:0", arg226_1: "bf16[2560][1]cuda:0", arg227_1: "bf16[2560, 2560][2560, 1]cuda:0", arg228_1: "bf16[2560][1]cuda:0", arg229_1: "bf16[2560, 2560][2560, 1]cuda:0", arg230_1: "bf16[2560][1]cuda:0", arg231_1: "bf16[2560][1]cuda:0", arg232_1: "bf16[2560][1]cuda:0", arg233_1: "bf16[2560, 2560][2560, 1]cuda:0", arg234_1: "bf16[2560][1]cuda:0", arg235_1: "bf16[2560, 2560][2560, 1]cuda:0", arg236_1: "bf16[2560][1]cuda:0", arg237_1: "bf16[2560, 2560][2560, 1]cuda:0", arg238_1: "bf16[2560][1]cuda:0", arg239_1: "bf16[2560, 2560][2560, 1]cuda:0", arg240_1: "bf16[2560][1]cuda:0", arg241_1: "bf16[2560][1]cuda:0", arg242_1: "bf16[2560][1]cuda:0", arg243_1: "bf16[10240, 2560][2560, 1]cuda:0", arg244_1: "bf16[10240][1]cuda:0", arg245_1: "bf16[2560, 10240][10240, 1]cuda:0", arg246_1: "bf16[2560][1]cuda:0", arg247_1: "bf16[2560][1]cuda:0", arg248_1: "bf16[2560][1]cuda:0", arg249_1: "bf16[2560, 2560][2560, 1]cuda:0", arg250_1: "bf16[2560][1]cuda:0", arg251_1: "bf16[2560, 2560][2560, 1]cuda:0", arg252_1: "bf16[2560][1]cuda:0", arg253_1: "bf16[2560, 2560][2560, 1]cuda:0", arg254_1: "bf16[2560][1]cuda:0", arg255_1: "bf16[2560, 2560][2560, 1]cuda:0", arg256_1: "bf16[2560][1]cuda:0", arg257_1: "bf16[2560][1]cuda:0", arg258_1: "bf16[2560][1]cuda:0", arg259_1: "bf16[2560, 2560][2560, 1]cuda:0", arg260_1: "bf16[2560][1]cuda:0", arg261_1: "bf16[2560, 2560][2560, 1]cuda:0", arg262_1: "bf16[2560][1]cuda:0", arg263_1: "bf16[2560, 2560][2560, 1]cuda:0", arg264_1: "bf16[2560][1]cuda:0", arg265_1: "bf16[2560, 2560][2560, 1]cuda:0", arg266_1: "bf16[2560][1]cuda:0", arg267_1: "bf16[2560][1]cuda:0", arg268_1: "bf16[2560][1]cuda:0", arg269_1: "bf16[10240, 2560][2560, 1]cuda:0", arg270_1: "bf16[10240][1]cuda:0", arg271_1: "bf16[2560, 10240][10240, 1]cuda:0", arg272_1: "bf16[2560][1]cuda:0", arg273_1: "bf16[2560][1]cuda:0", arg274_1: "bf16[2560][1]cuda:0", arg275_1: "bf16[2560, 2560][2560, 1]cuda:0", arg276_1: "bf16[2560][1]cuda:0", arg277_1: "bf16[2560, 2560][2560, 1]cuda:0", arg278_1: "bf16[2560][1]cuda:0", arg279_1: "bf16[2560, 2560][2560, 1]cuda:0", arg280_1: "bf16[2560][1]cuda:0", arg281_1: "bf16[2560, 2560][2560, 1]cuda:0", arg282_1: "bf16[2560][1]cuda:0", arg283_1: "bf16[2560][1]cuda:0", arg284_1: "bf16[2560][1]cuda:0", arg285_1: "bf16[2560, 2560][2560, 1]cuda:0", arg286_1: "bf16[2560][1]cuda:0", arg287_1: "bf16[2560, 2560][2560, 1]cuda:0", arg288_1: "bf16[2560][1]cuda:0", arg289_1: "bf16[2560, 2560][2560, 1]cuda:0", arg290_1: "bf16[2560][1]cuda:0", arg291_1: "bf16[2560, 2560][2560, 1]cuda:0", arg292_1: "bf16[2560][1]cuda:0", arg293_1: "bf16[2560][1]cuda:0", arg294_1: "bf16[2560][1]cuda:0", arg295_1: "bf16[10240, 2560][2560, 1]cuda:0", arg296_1: "bf16[10240][1]cuda:0", arg297_1: "bf16[2560, 10240][10240, 1]cuda:0", arg298_1: "bf16[2560][1]cuda:0", arg299_1: "bf16[2560][1]cuda:0", arg300_1: "bf16[2560][1]cuda:0", arg301_1: "bf16[2560, 2560][2560, 1]cuda:0", arg302_1: "bf16[2560][1]cuda:0", arg303_1: "bf16[2560, 2560][2560, 1]cuda:0", arg304_1: "bf16[2560][1]cuda:0", arg305_1: "bf16[2560, 2560][2560, 1]cuda:0", arg306_1: "bf16[2560][1]cuda:0", arg307_1: "bf16[2560, 2560][2560, 1]cuda:0", arg308_1: "bf16[2560][1]cuda:0", arg309_1: "bf16[2560][1]cuda:0", arg310_1: "bf16[2560][1]cuda:0", arg311_1: "bf16[2560, 2560][2560, 1]cuda:0", arg312_1: "bf16[2560][1]cuda:0", arg313_1: "bf16[2560, 2560][2560, 1]cuda:0", arg314_1: "bf16[2560][1]cuda:0", arg315_1: "bf16[2560, 2560][2560, 1]cuda:0", arg316_1: "bf16[2560][1]cuda:0", arg317_1: "bf16[2560, 2560][2560, 1]cuda:0", arg318_1: "bf16[2560][1]cuda:0", arg319_1: "bf16[2560][1]cuda:0", arg320_1: "bf16[2560][1]cuda:0", arg321_1: "bf16[10240, 2560][2560, 1]cuda:0", arg322_1: "bf16[10240][1]cuda:0", arg323_1: "bf16[2560, 10240][10240, 1]cuda:0", arg324_1: "bf16[2560][1]cuda:0", arg325_1: "bf16[2560][1]cuda:0", arg326_1: "bf16[2560][1]cuda:0", arg327_1: "bf16[2560, 2560][2560, 1]cuda:0", arg328_1: "bf16[2560][1]cuda:0", arg329_1: "bf16[2560, 2560][2560, 1]cuda:0", arg330_1: "bf16[2560][1]cuda:0", arg331_1: "bf16[2560, 2560][2560, 1]cuda:0", arg332_1: "bf16[2560][1]cuda:0", arg333_1: "bf16[2560, 2560][2560, 1]cuda:0", arg334_1: "bf16[2560][1]cuda:0", arg335_1: "bf16[2560][1]cuda:0", arg336_1: "bf16[2560][1]cuda:0", arg337_1: "bf16[2560, 2560][2560, 1]cuda:0", arg338_1: "bf16[2560][1]cuda:0", arg339_1: "bf16[2560, 2560][2560, 1]cuda:0", arg340_1: "bf16[2560][1]cuda:0", arg341_1: "bf16[2560, 2560][2560, 1]cuda:0", arg342_1: "bf16[2560][1]cuda:0", arg343_1: "bf16[2560, 2560][2560, 1]cuda:0", arg344_1: "bf16[2560][1]cuda:0", arg345_1: "bf16[2560][1]cuda:0", arg346_1: "bf16[2560][1]cuda:0", arg347_1: "bf16[10240, 2560][2560, 1]cuda:0", arg348_1: "bf16[10240][1]cuda:0", arg349_1: "bf16[2560, 10240][10240, 1]cuda:0", arg350_1: "bf16[2560][1]cuda:0", arg351_1: "bf16[2560][1]cuda:0", arg352_1: "bf16[2560][1]cuda:0", arg353_1: "bf16[2560, 2560][2560, 1]cuda:0", arg354_1: "bf16[2560][1]cuda:0", arg355_1: "bf16[2560, 2560][2560, 1]cuda:0", arg356_1: "bf16[2560][1]cuda:0", arg357_1: "bf16[2560, 2560][2560, 1]cuda:0", arg358_1: "bf16[2560][1]cuda:0", arg359_1: "bf16[2560, 2560][2560, 1]cuda:0", arg360_1: "bf16[2560][1]cuda:0", arg361_1: "bf16[2560][1]cuda:0", arg362_1: "bf16[2560][1]cuda:0", arg363_1: "bf16[2560, 2560][2560, 1]cuda:0", arg364_1: "bf16[2560][1]cuda:0", arg365_1: "bf16[2560, 2560][2560, 1]cuda:0", arg366_1: "bf16[2560][1]cuda:0", arg367_1: "bf16[2560, 2560][2560, 1]cuda:0", arg368_1: "bf16[2560][1]cuda:0", arg369_1: "bf16[2560, 2560][2560, 1]cuda:0", arg370_1: "bf16[2560][1]cuda:0", arg371_1: "bf16[2560][1]cuda:0", arg372_1: "bf16[2560][1]cuda:0", arg373_1: "bf16[10240, 2560][2560, 1]cuda:0", arg374_1: "bf16[10240][1]cuda:0", arg375_1: "bf16[2560, 10240][10240, 1]cuda:0", arg376_1: "bf16[2560][1]cuda:0", arg377_1: "bf16[2560][1]cuda:0", arg378_1: "bf16[2560][1]cuda:0", arg379_1: "bf16[2560, 2560][2560, 1]cuda:0", arg380_1: "bf16[2560][1]cuda:0", arg381_1: "bf16[2560, 2560][2560, 1]cuda:0", arg382_1: "bf16[2560][1]cuda:0", arg383_1: "bf16[2560, 2560][2560, 1]cuda:0", arg384_1: "bf16[2560][1]cuda:0", arg385_1: "bf16[2560, 2560][2560, 1]cuda:0", arg386_1: "bf16[2560][1]cuda:0", arg387_1: "bf16[2560][1]cuda:0", arg388_1: "bf16[2560][1]cuda:0", arg389_1: "bf16[2560, 2560][2560, 1]cuda:0", arg390_1: "bf16[2560][1]cuda:0", arg391_1: "bf16[2560, 2560][2560, 1]cuda:0", arg392_1: "bf16[2560][1]cuda:0", arg393_1: "bf16[2560, 2560][2560, 1]cuda:0", arg394_1: "bf16[2560][1]cuda:0", arg395_1: "bf16[2560, 2560][2560, 1]cuda:0", arg396_1: "bf16[2560][1]cuda:0", arg397_1: "bf16[2560][1]cuda:0", arg398_1: "bf16[2560][1]cuda:0", arg399_1: "bf16[10240, 2560][2560, 1]cuda:0", arg400_1: "bf16[10240][1]cuda:0", arg401_1: "bf16[2560, 10240][10240, 1]cuda:0", arg402_1: "bf16[2560][1]cuda:0", arg403_1: "bf16[2560][1]cuda:0", arg404_1: "bf16[2560][1]cuda:0", arg405_1: "bf16[2560, 2560][2560, 1]cuda:0", arg406_1: "bf16[2560][1]cuda:0", arg407_1: "bf16[2560, 2560][2560, 1]cuda:0", arg408_1: "bf16[2560][1]cuda:0", arg409_1: "bf16[2560, 2560][2560, 1]cuda:0", arg410_1: "bf16[2560][1]cuda:0", arg411_1: "bf16[2560, 2560][2560, 1]cuda:0", arg412_1: "bf16[2560][1]cuda:0", arg413_1: "bf16[2560][1]cuda:0", arg414_1: "bf16[2560][1]cuda:0", arg415_1: "bf16[2560, 2560][2560, 1]cuda:0", arg416_1: "bf16[2560][1]cuda:0", arg417_1: "bf16[2560, 2560][2560, 1]cuda:0", arg418_1: "bf16[2560][1]cuda:0", arg419_1: "bf16[2560, 2560][2560, 1]cuda:0", arg420_1: "bf16[2560][1]cuda:0", arg421_1: "bf16[2560, 2560][2560, 1]cuda:0", arg422_1: "bf16[2560][1]cuda:0", arg423_1: "bf16[2560][1]cuda:0", arg424_1: "bf16[2560][1]cuda:0", arg425_1: "bf16[10240, 2560][2560, 1]cuda:0", arg426_1: "bf16[10240][1]cuda:0", arg427_1: "bf16[2560, 10240][10240, 1]cuda:0", arg428_1: "bf16[2560][1]cuda:0", arg429_1: "bf16[2560][1]cuda:0", arg430_1: "bf16[2560][1]cuda:0", arg431_1: "bf16[2560, 2560][2560, 1]cuda:0", arg432_1: "bf16[2560][1]cuda:0", arg433_1: "bf16[2560, 2560][2560, 1]cuda:0", arg434_1: "bf16[2560][1]cuda:0", arg435_1: "bf16[2560, 2560][2560, 1]cuda:0", arg436_1: "bf16[2560][1]cuda:0", arg437_1: "bf16[2560, 2560][2560, 1]cuda:0", arg438_1: "bf16[2560][1]cuda:0", arg439_1: "bf16[2560][1]cuda:0", arg440_1: "bf16[2560][1]cuda:0", arg441_1: "bf16[2560, 2560][2560, 1]cuda:0", arg442_1: "bf16[2560][1]cuda:0", arg443_1: "bf16[2560, 2560][2560, 1]cuda:0", arg444_1: "bf16[2560][1]cuda:0", arg445_1: "bf16[2560, 2560][2560, 1]cuda:0", arg446_1: "bf16[2560][1]cuda:0", arg447_1: "bf16[2560, 2560][2560, 1]cuda:0", arg448_1: "bf16[2560][1]cuda:0", arg449_1: "bf16[2560][1]cuda:0", arg450_1: "bf16[2560][1]cuda:0", arg451_1: "bf16[10240, 2560][2560, 1]cuda:0", arg452_1: "bf16[10240][1]cuda:0", arg453_1: "bf16[2560, 10240][10240, 1]cuda:0", arg454_1: "bf16[2560][1]cuda:0", arg455_1: "bf16[2560][1]cuda:0", arg456_1: "bf16[2560][1]cuda:0", arg457_1: "bf16[2560, 2560][2560, 1]cuda:0", arg458_1: "bf16[2560][1]cuda:0", arg459_1: "bf16[2560, 2560][2560, 1]cuda:0", arg460_1: "bf16[2560][1]cuda:0", arg461_1: "bf16[2560, 2560][2560, 1]cuda:0", arg462_1: "bf16[2560][1]cuda:0", arg463_1: "bf16[2560, 2560][2560, 1]cuda:0", arg464_1: "bf16[2560][1]cuda:0", arg465_1: "bf16[2560][1]cuda:0", arg466_1: "bf16[2560][1]cuda:0", arg467_1: "bf16[2560, 2560][2560, 1]cuda:0", arg468_1: "bf16[2560][1]cuda:0", arg469_1: "bf16[2560, 2560][2560, 1]cuda:0", arg470_1: "bf16[2560][1]cuda:0", arg471_1: "bf16[2560, 2560][2560, 1]cuda:0", arg472_1: "bf16[2560][1]cuda:0", arg473_1: "bf16[2560, 2560][2560, 1]cuda:0", arg474_1: "bf16[2560][1]cuda:0", arg475_1: "bf16[2560][1]cuda:0", arg476_1: "bf16[2560][1]cuda:0", arg477_1: "bf16[10240, 2560][2560, 1]cuda:0", arg478_1: "bf16[10240][1]cuda:0", arg479_1: "bf16[2560, 10240][10240, 1]cuda:0", arg480_1: "bf16[2560][1]cuda:0", arg481_1: "bf16[2560][1]cuda:0", arg482_1: "bf16[2560][1]cuda:0", arg483_1: "bf16[2560, 2560][2560, 1]cuda:0", arg484_1: "bf16[2560][1]cuda:0", arg485_1: "bf16[2560, 2560][2560, 1]cuda:0", arg486_1: "bf16[2560][1]cuda:0", arg487_1: "bf16[2560, 2560][2560, 1]cuda:0", arg488_1: "bf16[2560][1]cuda:0", arg489_1: "bf16[2560, 2560][2560, 1]cuda:0", arg490_1: "bf16[2560][1]cuda:0", arg491_1: "bf16[2560][1]cuda:0", arg492_1: "bf16[2560][1]cuda:0", arg493_1: "bf16[2560, 2560][2560, 1]cuda:0", arg494_1: "bf16[2560][1]cuda:0", arg495_1: "bf16[2560, 2560][2560, 1]cuda:0", arg496_1: "bf16[2560][1]cuda:0", arg497_1: "bf16[2560, 2560][2560, 1]cuda:0", arg498_1: "bf16[2560][1]cuda:0", arg499_1: "bf16[2560, 2560][2560, 1]cuda:0", arg500_1: "bf16[2560][1]cuda:0", arg501_1: "bf16[2560][1]cuda:0", arg502_1: "bf16[2560][1]cuda:0", arg503_1: "bf16[10240, 2560][2560, 1]cuda:0", arg504_1: "bf16[10240][1]cuda:0", arg505_1: "bf16[2560, 10240][10240, 1]cuda:0", arg506_1: "bf16[2560][1]cuda:0", arg507_1: "bf16[2560][1]cuda:0", arg508_1: "bf16[2560][1]cuda:0", arg509_1: "bf16[2560, 2560][2560, 1]cuda:0", arg510_1: "bf16[2560][1]cuda:0", arg511_1: "bf16[2560, 2560][2560, 1]cuda:0", arg512_1: "bf16[2560][1]cuda:0", arg513_1: "bf16[2560, 2560][2560, 1]cuda:0", arg514_1: "bf16[2560][1]cuda:0", arg515_1: "bf16[2560, 2560][2560, 1]cuda:0", arg516_1: "bf16[2560][1]cuda:0", arg517_1: "bf16[2560][1]cuda:0", arg518_1: "bf16[2560][1]cuda:0", arg519_1: "bf16[2560, 2560][2560, 1]cuda:0", arg520_1: "bf16[2560][1]cuda:0", arg521_1: "bf16[2560, 2560][2560, 1]cuda:0", arg522_1: "bf16[2560][1]cuda:0", arg523_1: "bf16[2560, 2560][2560, 1]cuda:0", arg524_1: "bf16[2560][1]cuda:0", arg525_1: "bf16[2560, 2560][2560, 1]cuda:0", arg526_1: "bf16[2560][1]cuda:0", arg527_1: "bf16[2560][1]cuda:0", arg528_1: "bf16[2560][1]cuda:0", arg529_1: "bf16[10240, 2560][2560, 1]cuda:0", arg530_1: "bf16[10240][1]cuda:0", arg531_1: "bf16[2560, 10240][10240, 1]cuda:0", arg532_1: "bf16[2560][1]cuda:0", arg533_1: "bf16[2560][1]cuda:0", arg534_1: "bf16[2560][1]cuda:0", arg535_1: "bf16[2560, 2560][2560, 1]cuda:0", arg536_1: "bf16[2560][1]cuda:0", arg537_1: "bf16[2560, 2560][2560, 1]cuda:0", arg538_1: "bf16[2560][1]cuda:0", arg539_1: "bf16[2560, 2560][2560, 1]cuda:0", arg540_1: "bf16[2560][1]cuda:0", arg541_1: "bf16[2560, 2560][2560, 1]cuda:0", arg542_1: "bf16[2560][1]cuda:0", arg543_1: "bf16[2560][1]cuda:0", arg544_1: "bf16[2560][1]cuda:0", arg545_1: "bf16[2560, 2560][2560, 1]cuda:0", arg546_1: "bf16[2560][1]cuda:0", arg547_1: "bf16[2560, 2560][2560, 1]cuda:0", arg548_1: "bf16[2560][1]cuda:0", arg549_1: "bf16[2560, 2560][2560, 1]cuda:0", arg550_1: "bf16[2560][1]cuda:0", arg551_1: "bf16[2560, 2560][2560, 1]cuda:0", arg552_1: "bf16[2560][1]cuda:0", arg553_1: "bf16[2560][1]cuda:0", arg554_1: "bf16[2560][1]cuda:0", arg555_1: "bf16[10240, 2560][2560, 1]cuda:0", arg556_1: "bf16[10240][1]cuda:0", arg557_1: "bf16[2560, 10240][10240, 1]cuda:0", arg558_1: "bf16[2560][1]cuda:0", arg559_1: "bf16[2560][1]cuda:0", arg560_1: "bf16[2560][1]cuda:0", arg561_1: "bf16[2560, 2560][2560, 1]cuda:0", arg562_1: "bf16[2560][1]cuda:0", arg563_1: "bf16[2560, 2560][2560, 1]cuda:0", arg564_1: "bf16[2560][1]cuda:0", arg565_1: "bf16[2560, 2560][2560, 1]cuda:0", arg566_1: "bf16[2560][1]cuda:0", arg567_1: "bf16[2560, 2560][2560, 1]cuda:0", arg568_1: "bf16[2560][1]cuda:0", arg569_1: "bf16[2560][1]cuda:0", arg570_1: "bf16[2560][1]cuda:0", arg571_1: "bf16[2560, 2560][2560, 1]cuda:0", arg572_1: "bf16[2560][1]cuda:0", arg573_1: "bf16[2560, 2560][2560, 1]cuda:0", arg574_1: "bf16[2560][1]cuda:0", arg575_1: "bf16[2560, 2560][2560, 1]cuda:0", arg576_1: "bf16[2560][1]cuda:0", arg577_1: "bf16[2560, 2560][2560, 1]cuda:0", arg578_1: "bf16[2560][1]cuda:0", arg579_1: "bf16[2560][1]cuda:0", arg580_1: "bf16[2560][1]cuda:0", arg581_1: "bf16[10240, 2560][2560, 1]cuda:0", arg582_1: "bf16[10240][1]cuda:0", arg583_1: "bf16[2560, 10240][10240, 1]cuda:0", arg584_1: "bf16[2560][1]cuda:0", arg585_1: "bf16[2560][1]cuda:0", arg586_1: "bf16[2560][1]cuda:0", arg587_1: "bf16[2560, 2560][2560, 1]cuda:0", arg588_1: "bf16[2560][1]cuda:0", arg589_1: "bf16[2560, 2560][2560, 1]cuda:0", arg590_1: "bf16[2560][1]cuda:0", arg591_1: "bf16[2560, 2560][2560, 1]cuda:0", arg592_1: "bf16[2560][1]cuda:0", arg593_1: "bf16[2560, 2560][2560, 1]cuda:0", arg594_1: "bf16[2560][1]cuda:0", arg595_1: "bf16[2560][1]cuda:0", arg596_1: "bf16[2560][1]cuda:0", arg597_1: "bf16[2560, 2560][2560, 1]cuda:0", arg598_1: "bf16[2560][1]cuda:0", arg599_1: "bf16[2560, 2560][2560, 1]cuda:0", arg600_1: "bf16[2560][1]cuda:0", arg601_1: "bf16[2560, 2560][2560, 1]cuda:0", arg602_1: "bf16[2560][1]cuda:0", arg603_1: "bf16[2560, 2560][2560, 1]cuda:0", arg604_1: "bf16[2560][1]cuda:0", arg605_1: "bf16[2560][1]cuda:0", arg606_1: "bf16[2560][1]cuda:0", arg607_1: "bf16[10240, 2560][2560, 1]cuda:0", arg608_1: "bf16[10240][1]cuda:0", arg609_1: "bf16[2560, 10240][10240, 1]cuda:0", arg610_1: "bf16[2560][1]cuda:0", arg611_1: "bf16[2560][1]cuda:0", arg612_1: "bf16[2560][1]cuda:0", arg613_1: "bf16[2560, 2560][2560, 1]cuda:0", arg614_1: "bf16[2560][1]cuda:0", arg615_1: "bf16[2560, 2560][2560, 1]cuda:0", arg616_1: "bf16[2560][1]cuda:0", arg617_1: "bf16[2560, 2560][2560, 1]cuda:0", arg618_1: "bf16[2560][1]cuda:0", arg619_1: "bf16[2560, 2560][2560, 1]cuda:0", arg620_1: "bf16[2560][1]cuda:0", arg621_1: "bf16[2560][1]cuda:0", arg622_1: "bf16[2560][1]cuda:0", arg623_1: "bf16[2560, 2560][2560, 1]cuda:0", arg624_1: "bf16[2560][1]cuda:0", arg625_1: "bf16[2560, 2560][2560, 1]cuda:0", arg626_1: "bf16[2560][1]cuda:0", arg627_1: "bf16[2560, 2560][2560, 1]cuda:0", arg628_1: "bf16[2560][1]cuda:0", arg629_1: "bf16[2560, 2560][2560, 1]cuda:0", arg630_1: "bf16[2560][1]cuda:0", arg631_1: "bf16[2560][1]cuda:0", arg632_1: "bf16[2560][1]cuda:0", arg633_1: "bf16[10240, 2560][2560, 1]cuda:0", arg634_1: "bf16[10240][1]cuda:0", arg635_1: "bf16[2560, 10240][10240, 1]cuda:0", arg636_1: "bf16[2560][1]cuda:0", arg637_1: "bf16[2560][1]cuda:0", arg638_1: "bf16[2560][1]cuda:0", arg639_1: "bf16[2560, 2560][2560, 1]cuda:0", arg640_1: "bf16[2560][1]cuda:0", arg641_1: "bf16[2560, 2560][2560, 1]cuda:0", arg642_1: "bf16[2560][1]cuda:0", arg643_1: "bf16[2560, 2560][2560, 1]cuda:0", arg644_1: "bf16[2560][1]cuda:0", arg645_1: "bf16[2560, 2560][2560, 1]cuda:0", arg646_1: "bf16[2560][1]cuda:0", arg647_1: "bf16[2560][1]cuda:0", arg648_1: "bf16[2560][1]cuda:0", arg649_1: "bf16[2560, 2560][2560, 1]cuda:0", arg650_1: "bf16[2560][1]cuda:0", arg651_1: "bf16[2560, 2560][2560, 1]cuda:0", arg652_1: "bf16[2560][1]cuda:0", arg653_1: "bf16[2560, 2560][2560, 1]cuda:0", arg654_1: "bf16[2560][1]cuda:0", arg655_1: "bf16[2560, 2560][2560, 1]cuda:0", arg656_1: "bf16[2560][1]cuda:0", arg657_1: "bf16[2560][1]cuda:0", arg658_1: "bf16[2560][1]cuda:0", arg659_1: "bf16[10240, 2560][2560, 1]cuda:0", arg660_1: "bf16[10240][1]cuda:0", arg661_1: "bf16[2560, 10240][10240, 1]cuda:0", arg662_1: "bf16[2560][1]cuda:0", arg663_1: "bf16[2560][1]cuda:0", arg664_1: "bf16[2560][1]cuda:0", arg665_1: "bf16[1, 8008][8008, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_1: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[16, 1, 128, 128][0, 128, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge, [16, -1, 128, 128]);  ge = expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:96 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 0)
        mul: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:79 in forward, code: position_ids = torch.arange(
        iota: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_1: "bf16[128, 2560][2560, 1]cuda:0" = torch.ops.aten.embedding.default(arg3_1, iota);  arg3_1 = iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:484 in forward, code: hidden_states = inputs_embeds + embed_pos
        add: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:279 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor = None
        scalar_tensor_1: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_1 = None
        full_default: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:279 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_3: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_1: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        add_4: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        convert_element_type_1: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [2048, 2560])
        permute: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [16, 128, 2560]);  addmm = None
        view_2: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [16, 128, -1, 80]);  view_1 = None
        permute_1: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_3: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_1, 0.334370152488211);  permute_1 = None
        expand_1: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_3, [16, 32, 128, 80]);  mul_3 = None
        clone_1: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [512, 128, 80]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_3: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [2048, 2560])
        permute_2: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg9_1, view_3, permute_2);  arg9_1 = view_3 = permute_2 = None
        view_4: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [16, 128, 2560]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [16, 128, -1, 80]);  view_4 = None
        permute_4: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_6: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        mul_4: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_6, 0.334370152488211);  permute_6 = None
        expand_2: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_4, [16, 32, 80, 128]);  mul_4 = None
        clone_2: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [512, 80, 128]);  clone_2 = None
        bmm: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [16, 32, 128, 128]);  bmm = None
        eq: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_11, -inf)
        logical_not: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_1: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_13: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        amax: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_1: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax);  convert_element_type_13 = amax = None
        exp: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_14: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where_1: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_1, convert_element_type_14);  logical_not_1 = full_default_1 = convert_element_type_14 = None
        expand_3: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_1, [16, 32, 128, 128]);  where_1 = None
        view_12: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_3, [512, 128, 128]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_5: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [2048, 2560]);  convert_element_type_1 = None
        permute_3: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg11_1, view_5, permute_3);  arg11_1 = view_5 = permute_3 = None
        view_6: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [16, 128, 2560]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [16, 128, -1, 80]);  view_6 = None
        permute_5: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [16, 32, 128, 80]);  permute_5 = None
        clone_3: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_13: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [512, 128, 80]);  clone_3 = None
        bmm_1: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
        view_14: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [16, 32, 128, 80]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_4: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [16, 128, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_16: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [2048, 2560]);  view_15 = None
        permute_8: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg13_1, view_16, permute_8);  arg13_1 = view_16 = permute_8 = None
        view_17: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [16, 128, 2560]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:286 in forward, code: hidden_states = residual + hidden_states
        add_6: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add, view_17);  add = view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:289 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_20: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_3);  convert_element_type_20 = getitem_3 = None
        add_7: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_5: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        add_8: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, arg15_1);  mul_6 = arg15_1 = None
        convert_element_type_21: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_18: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_21, [2048, 2560]);  convert_element_type_21 = None
        permute_9: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg17_1, view_18, permute_9);  arg17_1 = view_18 = permute_9 = None
        view_19: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [16, 128, 10240]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_25: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_7: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, 0.5)
        mul_8: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, 0.7071067811865476);  convert_element_type_25 = None
        erf: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_9: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
        convert_element_type_26: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        view_20: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_26, [2048, 10240]);  convert_element_type_26 = None
        permute_10: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg19_1, view_20, permute_10);  arg19_1 = view_20 = permute_10 = None
        view_21: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [16, 128, 2560]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:294 in forward, code: hidden_states = residual + hidden_states
        add_10: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_6, view_21);  add_6 = view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:279 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_30: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True)
        getitem_4: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_2: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_2 = None
        scalar_tensor_3: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_3 = None
        full_default_2: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:279 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_3: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_5);  convert_element_type_30 = getitem_5 = None
        add_11: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_10: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_11: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        add_12: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, arg21_1);  mul_11 = arg21_1 = None
        convert_element_type_31: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_22: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [2048, 2560])
        permute_11: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg23_1, view_22, permute_11);  arg23_1 = view_22 = permute_11 = None
        view_23: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [16, 128, 2560]);  addmm_6 = None
        view_24: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [16, 128, -1, 80]);  view_23 = None
        permute_12: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_12: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_12, 0.334370152488211);  permute_12 = None
        expand_5: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_12, [16, 32, 128, 80]);  mul_12 = None
        clone_8: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_31: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [512, 128, 80]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_25: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [2048, 2560])
        permute_13: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg25_1, view_25, permute_13);  arg25_1 = view_25 = permute_13 = None
        view_26: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [16, 128, 2560]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_29: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [16, 128, -1, 80]);  view_26 = None
        permute_15: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_17: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        mul_13: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_17, 0.334370152488211);  permute_17 = None
        expand_6: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_13, [16, 32, 80, 128]);  mul_13 = None
        clone_9: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_32: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [512, 80, 128]);  clone_9 = None
        bmm_2: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
        view_33: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [16, 32, 128, 128]);  bmm_2 = None
        eq_1: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_33, -inf)
        logical_not_2: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_3: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_43: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        amax_1: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_43, [-1], True)
        sub_4: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_43, amax_1);  convert_element_type_43 = amax_1 = None
        exp_1: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_44: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        where_3: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_3, full_default_3, convert_element_type_44);  logical_not_3 = full_default_3 = convert_element_type_44 = None
        expand_7: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_3, [16, 32, 128, 128]);  where_3 = None
        view_34: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_7, [512, 128, 128]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_27: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [2048, 2560]);  convert_element_type_31 = None
        permute_14: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg27_1, view_27, permute_14);  arg27_1 = view_27 = permute_14 = None
        view_28: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [16, 128, 2560]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_30: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_28, [16, 128, -1, 80]);  view_28 = None
        permute_16: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_16, [16, 32, 128, 80]);  permute_16 = None
        clone_10: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_35: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [512, 128, 80]);  clone_10 = None
        bmm_3: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [16, 32, 128, 80]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_11: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [16, 128, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_38: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [2048, 2560]);  view_37 = None
        permute_19: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg29_1, view_38, permute_19);  arg29_1 = view_38 = permute_19 = None
        view_39: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [16, 128, 2560]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:286 in forward, code: hidden_states = residual + hidden_states
        add_14: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, view_39);  add_10 = view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:289 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_50: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_6: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_7);  convert_element_type_50 = getitem_7 = None
        add_15: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_14: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_15: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, arg30_1);  mul_14 = arg30_1 = None
        add_16: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, arg31_1);  mul_15 = arg31_1 = None
        convert_element_type_51: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_40: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_51, [2048, 2560]);  convert_element_type_51 = None
        permute_20: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg33_1, view_40, permute_20);  arg33_1 = view_40 = permute_20 = None
        view_41: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [16, 128, 10240]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_55: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_16: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.5)
        mul_17: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.7071067811865476);  convert_element_type_55 = None
        erf_1: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_17: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_18: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, add_17);  mul_16 = add_17 = None
        convert_element_type_56: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        view_42: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_56, [2048, 10240]);  convert_element_type_56 = None
        permute_21: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg35_1, view_42, permute_21);  arg35_1 = view_42 = permute_21 = None
        view_43: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [16, 128, 2560]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:294 in forward, code: hidden_states = residual + hidden_states
        add_18: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_14, view_43);  add_14 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:508 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_60: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.float32);  add_18 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True)
        getitem_8: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_12: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_24: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_12, 0);  iota_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_9: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_24, 0);  add_24 = None
        unsqueeze_10: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        unsqueeze_11: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_1: "b8[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_11, 0);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_10: "b8[16, 1, 128, 128][0, 128, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge_1, [16, -1, 128, 128]);  ge_1 = expand_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:96 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_2: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 0);  arg1_1 = None
        mul_21: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:585 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota_5: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_21: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_5, 0);  iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_3: "bf16[128, 2560][2560, 1]cuda:0" = torch.ops.aten.embedding.default(arg38_1, add_21);  arg38_1 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:616 in forward, code: hidden_states = inputs_embeds + position_ids
        add_26: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, embedding_3);  mul_21 = embedding_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_62: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_62, [2], correction = 0, keepdim = True)
        getitem_10: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_11);  convert_element_type_62 = getitem_11 = None
        add_27: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_22: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_23: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg39_1);  mul_22 = arg39_1 = None
        add_28: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg40_1);  mul_23 = arg40_1 = None
        convert_element_type_63: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_44: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [2048, 2560])
        permute_22: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_12: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_44, permute_22);  arg42_1 = view_44 = permute_22 = None
        view_45: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [16, 128, 2560]);  addmm_12 = None
        view_46: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [16, 128, -1, 80]);  view_45 = None
        permute_23: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_47: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [2048, 2560])
        permute_24: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_13: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_47, permute_24);  arg44_1 = view_47 = permute_24 = None
        view_48: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [16, 128, 2560]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_51: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [16, 128, -1, 80]);  view_48 = None
        permute_26: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_49: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [2048, 2560]);  convert_element_type_63 = None
        permute_25: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_14: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg46_1, view_49, permute_25);  arg46_1 = view_49 = permute_25 = None
        view_50: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [16, 128, 2560]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_52: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [16, 128, -1, 80]);  view_50 = None
        permute_27: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_9: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_23: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_9, 0);  iota_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_6: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_23, 0);  add_23 = None
        unsqueeze_7: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_8: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_22: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_8, 0);  iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_3: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_22, 0);  add_22 = None
        unsqueeze_4: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_8, unsqueeze_5);  unsqueeze_8 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_9: "b8[16, 1, 128, 128][0, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(le, [16, -1, 128, 128]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_5: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_5, full_default_4);  full_default_5 = full_default_4 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_23, permute_26, permute_27, where_4, False, scale = 0.11180339887498948);  permute_23 = permute_26 = permute_27 = where_4 = None
        getitem_12: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_28: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_53: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_28, [16, 128, -1]);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_54: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_53, [2048, 2560]);  view_53 = None
        permute_29: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_15: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, view_54, permute_29);  arg48_1 = view_54 = permute_29 = None
        view_55: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [16, 128, 2560]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_29: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_26, view_55);  add_26 = view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_76: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_76, [2], correction = 0, keepdim = True)
        getitem_21: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_6[0]
        getitem_22: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_6: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_6 = None
        scalar_tensor_7: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_7 = None
        full_default_6: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_8: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_76, getitem_22);  convert_element_type_76 = getitem_22 = None
        add_30: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_21, 1e-05);  getitem_21 = None
        rsqrt_6: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_24: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_25: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, arg49_1);  mul_24 = arg49_1 = None
        add_31: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, arg50_1);  mul_25 = arg50_1 = None
        convert_element_type_77: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_56: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_77, [2048, 2560]);  convert_element_type_77 = None
        permute_30: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_16: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg52_1, view_56, permute_30);  arg52_1 = view_56 = permute_30 = None
        view_57: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [16, 128, 2560]);  addmm_16 = None
        view_58: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_57, [16, 128, -1, 80]);  view_57 = None
        permute_31: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_26: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_31, 0.334370152488211);  permute_31 = None
        expand_11: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_26, [16, 32, 128, 80]);  mul_26 = None
        clone_17: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_65: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [512, 128, 80]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:508 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_6: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_9);  convert_element_type_60 = getitem_9 = None
        add_19: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_19: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_20: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, arg36_1);  mul_19 = arg36_1 = None
        add_20: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, arg37_1);  mul_20 = arg37_1 = None
        convert_element_type_61: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_59: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_32: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_17: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg54_1, view_59, permute_32);  arg54_1 = view_59 = permute_32 = None
        view_60: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [16, 128, 2560]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_63: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_60, [16, 128, -1, 80]);  view_60 = None
        permute_34: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_36: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_34, [0, 1, 3, 2]);  permute_34 = None
        mul_27: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_36, 0.334370152488211);  permute_36 = None
        expand_12: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_27, [16, 32, 80, 128]);  mul_27 = None
        clone_18: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_66: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [512, 80, 128]);  clone_18 = None
        bmm_4: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_65, view_66);  view_65 = view_66 = None
        view_67: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [16, 32, 128, 128]);  bmm_4 = None
        eq_2: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_67, -inf)
        logical_not_4: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_7: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_89: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_67, torch.float32);  view_67 = None
        amax_2: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_89, [-1], True)
        sub_9: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_89, amax_2);  convert_element_type_89 = amax_2 = None
        exp_2: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_90: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        where_6: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_5, full_default_7, convert_element_type_90);  logical_not_5 = full_default_7 = convert_element_type_90 = None
        expand_13: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_6, [16, 32, 128, 128]);  where_6 = None
        view_68: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_13, [512, 128, 128]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_61: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_33: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_18: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg56_1, view_61, permute_33);  arg56_1 = view_61 = permute_33 = None
        view_62: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [16, 128, 2560]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_64: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [16, 128, -1, 80]);  view_62 = None
        permute_35: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_14: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_35, [16, 32, 128, 80]);  permute_35 = None
        clone_19: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_69: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [512, 128, 80]);  clone_19 = None
        bmm_5: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_68, view_69);  view_68 = view_69 = None
        view_70: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [16, 32, 128, 80]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_37: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_20: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_71: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [16, 128, -1]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_72: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_71, [2048, 2560]);  view_71 = None
        permute_38: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_19: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg58_1, view_72, permute_38);  arg58_1 = view_72 = permute_38 = None
        view_73: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [16, 128, 2560]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_33: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_29, view_73);  add_29 = view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_96: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_23: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_7[0]
        getitem_24: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_10: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_24);  convert_element_type_96 = getitem_24 = None
        add_34: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_23, 1e-05);  getitem_23 = None
        rsqrt_7: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_28: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = rsqrt_7 = None
        mul_29: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, arg59_1);  mul_28 = arg59_1 = None
        add_35: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, arg60_1);  mul_29 = arg60_1 = None
        convert_element_type_97: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_74: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [2048, 2560]);  convert_element_type_97 = None
        permute_39: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_20: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg62_1, view_74, permute_39);  arg62_1 = view_74 = permute_39 = None
        view_75: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [16, 128, 10240]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_101: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        mul_30: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_101, 0.5)
        mul_31: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_101, 0.7071067811865476);  convert_element_type_101 = None
        erf_2: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_36: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_32: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, add_36);  mul_30 = add_36 = None
        convert_element_type_102: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_76: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_102, [2048, 10240]);  convert_element_type_102 = None
        permute_40: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_21: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg64_1, view_76, permute_40);  arg64_1 = view_76 = permute_40 = None
        view_77: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [16, 128, 2560]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_37: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_33, view_77);  add_33 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_106: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_106, [2], correction = 0, keepdim = True)
        getitem_25: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_8[0]
        getitem_26: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_11: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_106, getitem_26);  convert_element_type_106 = getitem_26 = None
        add_38: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_25, 1e-05);  getitem_25 = None
        rsqrt_8: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_33: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = rsqrt_8 = None
        mul_34: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, arg65_1);  mul_33 = arg65_1 = None
        add_39: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, arg66_1);  mul_34 = arg66_1 = None
        convert_element_type_107: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_78: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_107, [2048, 2560])
        permute_41: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_22: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg68_1, view_78, permute_41);  arg68_1 = view_78 = permute_41 = None
        view_79: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [16, 128, 2560]);  addmm_22 = None
        view_80: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_79, [16, 128, -1, 80]);  view_79 = None
        permute_42: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_81: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_107, [2048, 2560])
        permute_43: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_23: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg70_1, view_81, permute_43);  arg70_1 = view_81 = permute_43 = None
        view_82: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [16, 128, 2560]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_85: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_82, [16, 128, -1, 80]);  view_82 = None
        permute_45: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_83: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_107, [2048, 2560]);  convert_element_type_107 = None
        permute_44: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_24: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_83, permute_44);  arg72_1 = view_83 = permute_44 = None
        view_84: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [16, 128, 2560]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_86: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [16, 128, -1, 80]);  view_84 = None
        permute_46: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_42, permute_45, permute_46, where_7, False, scale = 0.11180339887498948);  permute_42 = permute_45 = permute_46 = where_7 = None
        getitem_27: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_87: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_47, [16, 128, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_88: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_87, [2048, 2560]);  view_87 = None
        permute_48: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_25: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_88, permute_48);  arg74_1 = view_88 = permute_48 = None
        view_89: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [16, 128, 2560]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_40: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, view_89);  add_37 = view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_120: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_36: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_9[0]
        getitem_37: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_10: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_10 = None
        scalar_tensor_11: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_11 = None
        full_default_10: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_12: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_37);  convert_element_type_120 = getitem_37 = None
        add_41: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_9: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_35: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_9);  sub_12 = rsqrt_9 = None
        mul_36: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, arg75_1);  mul_35 = arg75_1 = None
        add_42: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, arg76_1);  mul_36 = arg76_1 = None
        convert_element_type_121: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_42, torch.bfloat16);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_90: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [2048, 2560]);  convert_element_type_121 = None
        permute_49: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_26: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg78_1, view_90, permute_49);  arg78_1 = view_90 = permute_49 = None
        view_91: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [16, 128, 2560]);  addmm_26 = None
        view_92: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_91, [16, 128, -1, 80]);  view_91 = None
        permute_50: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_37: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_50, 0.334370152488211);  permute_50 = None
        expand_15: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_37, [16, 32, 128, 80]);  mul_37 = None
        clone_25: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_99: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [512, 128, 80]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_93: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_51: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_27: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_93, permute_51);  arg80_1 = view_93 = permute_51 = None
        view_94: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [16, 128, 2560]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_97: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [16, 128, -1, 80]);  view_94 = None
        permute_53: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_55: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_53, [0, 1, 3, 2]);  permute_53 = None
        mul_38: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_55, 0.334370152488211);  permute_55 = None
        expand_16: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_38, [16, 32, 80, 128]);  mul_38 = None
        clone_26: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_100: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [512, 80, 128]);  clone_26 = None
        bmm_6: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
        view_101: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [16, 32, 128, 128]);  bmm_6 = None
        eq_3: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_101, -inf)
        logical_not_6: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_11: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_133: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None
        amax_3: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_133, [-1], True)
        sub_13: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, amax_3);  convert_element_type_133 = amax_3 = None
        exp_3: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_4: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_134: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        where_9: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_7, full_default_11, convert_element_type_134);  logical_not_7 = full_default_11 = convert_element_type_134 = None
        expand_17: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_9, [16, 32, 128, 128]);  where_9 = None
        view_102: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_17, [512, 128, 128]);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_95: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_52: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_28: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg82_1, view_95, permute_52);  arg82_1 = view_95 = permute_52 = None
        view_96: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [16, 128, 2560]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_98: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_96, [16, 128, -1, 80]);  view_96 = None
        permute_54: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_18: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_54, [16, 32, 128, 80]);  permute_54 = None
        clone_27: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_103: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [512, 128, 80]);  clone_27 = None
        bmm_7: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
        view_104: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [16, 32, 128, 80]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_28: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [16, 128, -1]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_106: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [2048, 2560]);  view_105 = None
        permute_57: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_29: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg84_1, view_106, permute_57);  arg84_1 = view_106 = permute_57 = None
        view_107: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [16, 128, 2560]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_44: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_40, view_107);  add_40 = view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_140: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_140, [2], correction = 0, keepdim = True)
        getitem_38: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_10[0]
        getitem_39: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_14: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_140, getitem_39);  convert_element_type_140 = getitem_39 = None
        add_45: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_10: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_39: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = rsqrt_10 = None
        mul_40: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, arg85_1);  mul_39 = arg85_1 = None
        add_46: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_40, arg86_1);  mul_40 = arg86_1 = None
        convert_element_type_141: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_108: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_141, [2048, 2560]);  convert_element_type_141 = None
        permute_58: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_30: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg88_1, view_108, permute_58);  arg88_1 = view_108 = permute_58 = None
        view_109: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [16, 128, 10240]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_145: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        mul_41: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, 0.5)
        mul_42: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, 0.7071067811865476);  convert_element_type_145 = None
        erf_3: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_47: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_43: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, add_47);  mul_41 = add_47 = None
        convert_element_type_146: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_110: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_146, [2048, 10240]);  convert_element_type_146 = None
        permute_59: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_31: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg90_1, view_110, permute_59);  arg90_1 = view_110 = permute_59 = None
        view_111: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [16, 128, 2560]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_48: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_44, view_111);  add_44 = view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_150: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_150, [2], correction = 0, keepdim = True)
        getitem_40: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_11[0]
        getitem_41: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_15: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_150, getitem_41);  convert_element_type_150 = getitem_41 = None
        add_49: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_11: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_44: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = rsqrt_11 = None
        mul_45: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, arg91_1);  mul_44 = arg91_1 = None
        add_50: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, arg92_1);  mul_45 = arg92_1 = None
        convert_element_type_151: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_112: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [2048, 2560])
        permute_60: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_32: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg94_1, view_112, permute_60);  arg94_1 = view_112 = permute_60 = None
        view_113: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [16, 128, 2560]);  addmm_32 = None
        view_114: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [16, 128, -1, 80]);  view_113 = None
        permute_61: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_115: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [2048, 2560])
        permute_62: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_33: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg96_1, view_115, permute_62);  arg96_1 = view_115 = permute_62 = None
        view_116: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [16, 128, 2560]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_119: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_116, [16, 128, -1, 80]);  view_116 = None
        permute_64: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_117: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [2048, 2560]);  convert_element_type_151 = None
        permute_63: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_34: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, view_117, permute_63);  arg98_1 = view_117 = permute_63 = None
        view_118: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [16, 128, 2560]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_120: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_118, [16, 128, -1, 80]);  view_118 = None
        permute_65: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_61, permute_64, permute_65, where_10, False, scale = 0.11180339887498948);  permute_61 = permute_64 = permute_65 = where_10 = None
        getitem_42: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_66: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_121: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_66, [16, 128, -1]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_122: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_121, [2048, 2560]);  view_121 = None
        permute_67: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_35: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg100_1, view_122, permute_67);  arg100_1 = view_122 = permute_67 = None
        view_123: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [16, 128, 2560]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_51: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_48, view_123);  add_48 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_164: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_164, [2], correction = 0, keepdim = True)
        getitem_51: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_12[0]
        getitem_52: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_14: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_14 = None
        scalar_tensor_15: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_15 = None
        full_default_14: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_16: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_164, getitem_52);  convert_element_type_164 = getitem_52 = None
        add_52: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_51, 1e-05);  getitem_51 = None
        rsqrt_12: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_46: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_12);  sub_16 = rsqrt_12 = None
        mul_47: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, arg101_1);  mul_46 = arg101_1 = None
        add_53: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_47, arg102_1);  mul_47 = arg102_1 = None
        convert_element_type_165: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_124: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_165, [2048, 2560]);  convert_element_type_165 = None
        permute_68: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_36: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg104_1, view_124, permute_68);  arg104_1 = view_124 = permute_68 = None
        view_125: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [16, 128, 2560]);  addmm_36 = None
        view_126: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [16, 128, -1, 80]);  view_125 = None
        permute_69: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_48: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_69, 0.334370152488211);  permute_69 = None
        expand_19: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_48, [16, 32, 128, 80]);  mul_48 = None
        clone_33: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_133: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [512, 128, 80]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_127: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_70: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_37: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg106_1, view_127, permute_70);  arg106_1 = view_127 = permute_70 = None
        view_128: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [16, 128, 2560]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_131: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_128, [16, 128, -1, 80]);  view_128 = None
        permute_72: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_74: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_72, [0, 1, 3, 2]);  permute_72 = None
        mul_49: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_74, 0.334370152488211);  permute_74 = None
        expand_20: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_49, [16, 32, 80, 128]);  mul_49 = None
        clone_34: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_134: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [512, 80, 128]);  clone_34 = None
        bmm_8: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_133, view_134);  view_133 = view_134 = None
        view_135: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [16, 32, 128, 128]);  bmm_8 = None
        eq_4: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_135, -inf)
        logical_not_8: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_15: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_177: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_135, torch.float32);  view_135 = None
        amax_4: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_177, [-1], True)
        sub_17: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_177, amax_4);  convert_element_type_177 = amax_4 = None
        exp_4: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_5: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_178: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        where_12: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_9, full_default_15, convert_element_type_178);  logical_not_9 = full_default_15 = convert_element_type_178 = None
        expand_21: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_12, [16, 32, 128, 128]);  where_12 = None
        view_136: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_21, [512, 128, 128]);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_129: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_71: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_38: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg108_1, view_129, permute_71);  arg108_1 = view_129 = permute_71 = None
        view_130: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [16, 128, 2560]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_132: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_130, [16, 128, -1, 80]);  view_130 = None
        permute_73: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_22: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_73, [16, 32, 128, 80]);  permute_73 = None
        clone_35: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_137: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [512, 128, 80]);  clone_35 = None
        bmm_9: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_136, view_137);  view_136 = view_137 = None
        view_138: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [16, 32, 128, 80]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_75: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_36: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_139: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [16, 128, -1]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_140: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_139, [2048, 2560]);  view_139 = None
        permute_76: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_39: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg110_1, view_140, permute_76);  arg110_1 = view_140 = permute_76 = None
        view_141: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [16, 128, 2560]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_55: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_51, view_141);  add_51 = view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_184: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_184, [2], correction = 0, keepdim = True)
        getitem_53: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_13[0]
        getitem_54: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_18: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_184, getitem_54);  convert_element_type_184 = getitem_54 = None
        add_56: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_13: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_50: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = rsqrt_13 = None
        mul_51: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, arg111_1);  mul_50 = arg111_1 = None
        add_57: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, arg112_1);  mul_51 = arg112_1 = None
        convert_element_type_185: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_142: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_185, [2048, 2560]);  convert_element_type_185 = None
        permute_77: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_40: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg114_1, view_142, permute_77);  arg114_1 = view_142 = permute_77 = None
        view_143: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [16, 128, 10240]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_189: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        mul_52: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_189, 0.5)
        mul_53: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_189, 0.7071067811865476);  convert_element_type_189 = None
        erf_4: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
        add_58: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_54: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, add_58);  mul_52 = add_58 = None
        convert_element_type_190: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_144: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_190, [2048, 10240]);  convert_element_type_190 = None
        permute_78: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_41: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg116_1, view_144, permute_78);  arg116_1 = view_144 = permute_78 = None
        view_145: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [16, 128, 2560]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_59: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_55, view_145);  add_55 = view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_194: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_194, [2], correction = 0, keepdim = True)
        getitem_55: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_14[0]
        getitem_56: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_19: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_194, getitem_56);  convert_element_type_194 = getitem_56 = None
        add_60: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_14: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_55: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_14);  sub_19 = rsqrt_14 = None
        mul_56: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, arg117_1);  mul_55 = arg117_1 = None
        add_61: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, arg118_1);  mul_56 = arg118_1 = None
        convert_element_type_195: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_146: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_195, [2048, 2560])
        permute_79: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_42: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg120_1, view_146, permute_79);  arg120_1 = view_146 = permute_79 = None
        view_147: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [16, 128, 2560]);  addmm_42 = None
        view_148: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [16, 128, -1, 80]);  view_147 = None
        permute_80: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_149: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_195, [2048, 2560])
        permute_81: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_43: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg122_1, view_149, permute_81);  arg122_1 = view_149 = permute_81 = None
        view_150: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [16, 128, 2560]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_153: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_150, [16, 128, -1, 80]);  view_150 = None
        permute_83: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_151: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_195, [2048, 2560]);  convert_element_type_195 = None
        permute_82: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_44: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg124_1, view_151, permute_82);  arg124_1 = view_151 = permute_82 = None
        view_152: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [16, 128, 2560]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_154: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_152, [16, 128, -1, 80]);  view_152 = None
        permute_84: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_80, permute_83, permute_84, where_13, False, scale = 0.11180339887498948);  permute_80 = permute_83 = permute_84 = where_13 = None
        getitem_57: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_85: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_155: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_85, [16, 128, -1]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_156: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [2048, 2560]);  view_155 = None
        permute_86: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_45: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg126_1, view_156, permute_86);  arg126_1 = view_156 = permute_86 = None
        view_157: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [16, 128, 2560]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_62: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_59, view_157);  add_59 = view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_208: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_208, [2], correction = 0, keepdim = True)
        getitem_66: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_15[0]
        getitem_67: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_18: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_18 = None
        scalar_tensor_19: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_19 = None
        full_default_18: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_20: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_208, getitem_67);  convert_element_type_208 = getitem_67 = None
        add_63: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_15: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_57: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_15);  sub_20 = rsqrt_15 = None
        mul_58: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg127_1);  mul_57 = arg127_1 = None
        add_64: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg128_1);  mul_58 = arg128_1 = None
        convert_element_type_209: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_158: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_209, [2048, 2560]);  convert_element_type_209 = None
        permute_87: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_46: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg130_1, view_158, permute_87);  arg130_1 = view_158 = permute_87 = None
        view_159: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [16, 128, 2560]);  addmm_46 = None
        view_160: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_159, [16, 128, -1, 80]);  view_159 = None
        permute_88: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_59: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_88, 0.334370152488211);  permute_88 = None
        expand_23: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_59, [16, 32, 128, 80]);  mul_59 = None
        clone_41: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_167: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [512, 128, 80]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_161: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_89: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_47: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg132_1, view_161, permute_89);  arg132_1 = view_161 = permute_89 = None
        view_162: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [16, 128, 2560]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_165: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_162, [16, 128, -1, 80]);  view_162 = None
        permute_91: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_93: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        mul_60: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_93, 0.334370152488211);  permute_93 = None
        expand_24: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_60, [16, 32, 80, 128]);  mul_60 = None
        clone_42: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_168: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [512, 80, 128]);  clone_42 = None
        bmm_10: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_167, view_168);  view_167 = view_168 = None
        view_169: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [16, 32, 128, 128]);  bmm_10 = None
        eq_5: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_169, -inf)
        logical_not_10: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_19: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_221: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_169, torch.float32);  view_169 = None
        amax_5: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_221, [-1], True)
        sub_21: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_221, amax_5);  convert_element_type_221 = amax_5 = None
        exp_5: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_6: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_222: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        where_15: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_11, full_default_19, convert_element_type_222);  logical_not_11 = full_default_19 = convert_element_type_222 = None
        expand_25: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_15, [16, 32, 128, 128]);  where_15 = None
        view_170: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_25, [512, 128, 128]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_163: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_90: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg134_1, view_163, permute_90);  arg134_1 = view_163 = permute_90 = None
        view_164: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [16, 128, 2560]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_166: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_164, [16, 128, -1, 80]);  view_164 = None
        permute_92: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_26: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_92, [16, 32, 128, 80]);  permute_92 = None
        clone_43: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_171: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [512, 128, 80]);  clone_43 = None
        bmm_11: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_170, view_171);  view_170 = view_171 = None
        view_172: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [16, 32, 128, 80]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_94: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        clone_44: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_173: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [16, 128, -1]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_174: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_173, [2048, 2560]);  view_173 = None
        permute_95: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg136_1, view_174, permute_95);  arg136_1 = view_174 = permute_95 = None
        view_175: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [16, 128, 2560]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_66: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62, view_175);  add_62 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_228: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_228, [2], correction = 0, keepdim = True)
        getitem_68: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_16[0]
        getitem_69: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_22: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_228, getitem_69);  convert_element_type_228 = getitem_69 = None
        add_67: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_16: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_61: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_16);  sub_22 = rsqrt_16 = None
        mul_62: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, arg137_1);  mul_61 = arg137_1 = None
        add_68: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, arg138_1);  mul_62 = arg138_1 = None
        convert_element_type_229: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_176: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_229, [2048, 2560]);  convert_element_type_229 = None
        permute_96: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_50: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_176, permute_96);  arg140_1 = view_176 = permute_96 = None
        view_177: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [16, 128, 10240]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_233: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        mul_63: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_233, 0.5)
        mul_64: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_233, 0.7071067811865476);  convert_element_type_233 = None
        erf_5: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_69: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_65: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, add_69);  mul_63 = add_69 = None
        convert_element_type_234: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_178: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_234, [2048, 10240]);  convert_element_type_234 = None
        permute_97: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_51: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg142_1, view_178, permute_97);  arg142_1 = view_178 = permute_97 = None
        view_179: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [16, 128, 2560]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_70: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_66, view_179);  add_66 = view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_238: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_238, [2], correction = 0, keepdim = True)
        getitem_70: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_17[0]
        getitem_71: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_23: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_238, getitem_71);  convert_element_type_238 = getitem_71 = None
        add_71: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_17: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_66: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_17);  sub_23 = rsqrt_17 = None
        mul_67: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg143_1);  mul_66 = arg143_1 = None
        add_72: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg144_1);  mul_67 = arg144_1 = None
        convert_element_type_239: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_180: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_239, [2048, 2560])
        permute_98: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_52: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg146_1, view_180, permute_98);  arg146_1 = view_180 = permute_98 = None
        view_181: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [16, 128, 2560]);  addmm_52 = None
        view_182: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_181, [16, 128, -1, 80]);  view_181 = None
        permute_99: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_183: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_239, [2048, 2560])
        permute_100: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_53: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg148_1, view_183, permute_100);  arg148_1 = view_183 = permute_100 = None
        view_184: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [16, 128, 2560]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_187: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_184, [16, 128, -1, 80]);  view_184 = None
        permute_102: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_185: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_239, [2048, 2560]);  convert_element_type_239 = None
        permute_101: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg150_1, view_185, permute_101);  arg150_1 = view_185 = permute_101 = None
        view_186: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [16, 128, 2560]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_188: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_186, [16, 128, -1, 80]);  view_186 = None
        permute_103: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_21: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_21, full_default_20);  full_default_21 = full_default_20 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_99, permute_102, permute_103, where_16, False, scale = 0.11180339887498948);  permute_99 = permute_102 = permute_103 = where_16 = None
        getitem_72: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_104: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_189: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_104, [16, 128, -1]);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_190: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_189, [2048, 2560]);  view_189 = None
        permute_105: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg152_1, view_190, permute_105);  arg152_1 = view_190 = permute_105 = None
        view_191: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [16, 128, 2560]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_73: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_70, view_191);  add_70 = view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_252: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_252, [2], correction = 0, keepdim = True)
        getitem_81: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_18[0]
        getitem_82: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_22: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_22 = None
        scalar_tensor_23: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_23 = None
        full_default_22: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_24: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_252, getitem_82);  convert_element_type_252 = getitem_82 = None
        add_74: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_81, 1e-05);  getitem_81 = None
        rsqrt_18: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_68: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_18);  sub_24 = rsqrt_18 = None
        mul_69: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, arg153_1);  mul_68 = arg153_1 = None
        add_75: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, arg154_1);  mul_69 = arg154_1 = None
        convert_element_type_253: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_192: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_253, [2048, 2560]);  convert_element_type_253 = None
        permute_106: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_56: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg156_1, view_192, permute_106);  arg156_1 = view_192 = permute_106 = None
        view_193: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [16, 128, 2560]);  addmm_56 = None
        view_194: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_193, [16, 128, -1, 80]);  view_193 = None
        permute_107: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_70: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_107, 0.334370152488211);  permute_107 = None
        expand_27: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_70, [16, 32, 128, 80]);  mul_70 = None
        clone_49: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_201: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [512, 128, 80]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_195: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_108: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_57: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg158_1, view_195, permute_108);  arg158_1 = view_195 = permute_108 = None
        view_196: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [16, 128, 2560]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_199: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_196, [16, 128, -1, 80]);  view_196 = None
        permute_110: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_112: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2]);  permute_110 = None
        mul_71: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_112, 0.334370152488211);  permute_112 = None
        expand_28: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_71, [16, 32, 80, 128]);  mul_71 = None
        clone_50: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_202: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [512, 80, 128]);  clone_50 = None
        bmm_12: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_201, view_202);  view_201 = view_202 = None
        view_203: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [16, 32, 128, 128]);  bmm_12 = None
        eq_6: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_203, -inf)
        logical_not_12: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_default_23: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_265: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_203, torch.float32);  view_203 = None
        amax_6: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_265, [-1], True)
        sub_25: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_265, amax_6);  convert_element_type_265 = amax_6 = None
        exp_6: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_7: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_266: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        where_18: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_13, full_default_23, convert_element_type_266);  logical_not_13 = full_default_23 = convert_element_type_266 = None
        expand_29: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_18, [16, 32, 128, 128]);  where_18 = None
        view_204: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_29, [512, 128, 128]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_197: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_109: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_197, permute_109);  arg160_1 = view_197 = permute_109 = None
        view_198: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [16, 128, 2560]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_200: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_198, [16, 128, -1, 80]);  view_198 = None
        permute_111: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_30: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_111, [16, 32, 128, 80]);  permute_111 = None
        clone_51: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_205: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [512, 128, 80]);  clone_51 = None
        bmm_13: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
        view_206: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [16, 32, 128, 80]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_113: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        clone_52: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_207: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [16, 128, -1]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_208: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_207, [2048, 2560]);  view_207 = None
        permute_114: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg162_1, view_208, permute_114);  arg162_1 = view_208 = permute_114 = None
        view_209: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [16, 128, 2560]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_77: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_73, view_209);  add_73 = view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_272: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_272, [2], correction = 0, keepdim = True)
        getitem_83: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_19[0]
        getitem_84: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_26: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, getitem_84);  convert_element_type_272 = getitem_84 = None
        add_78: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_83, 1e-05);  getitem_83 = None
        rsqrt_19: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_72: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_19);  sub_26 = rsqrt_19 = None
        mul_73: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, arg163_1);  mul_72 = arg163_1 = None
        add_79: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, arg164_1);  mul_73 = arg164_1 = None
        convert_element_type_273: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_210: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_273, [2048, 2560]);  convert_element_type_273 = None
        permute_115: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg166_1, view_210, permute_115);  arg166_1 = view_210 = permute_115 = None
        view_211: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [16, 128, 10240]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_277: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_211, torch.float32);  view_211 = None
        mul_74: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_277, 0.5)
        mul_75: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_277, 0.7071067811865476);  convert_element_type_277 = None
        erf_6: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_80: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_76: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, add_80);  mul_74 = add_80 = None
        convert_element_type_278: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_212: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_278, [2048, 10240]);  convert_element_type_278 = None
        permute_116: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg168_1, view_212, permute_116);  arg168_1 = view_212 = permute_116 = None
        view_213: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [16, 128, 2560]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_81: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_77, view_213);  add_77 = view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_282: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_282, [2], correction = 0, keepdim = True)
        getitem_85: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_20[0]
        getitem_86: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_27: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_282, getitem_86);  convert_element_type_282 = getitem_86 = None
        add_82: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_85, 1e-05);  getitem_85 = None
        rsqrt_20: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_77: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_20);  sub_27 = rsqrt_20 = None
        mul_78: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, arg169_1);  mul_77 = arg169_1 = None
        add_83: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, arg170_1);  mul_78 = arg170_1 = None
        convert_element_type_283: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_214: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_283, [2048, 2560])
        permute_117: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_62: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg172_1, view_214, permute_117);  arg172_1 = view_214 = permute_117 = None
        view_215: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [16, 128, 2560]);  addmm_62 = None
        view_216: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_215, [16, 128, -1, 80]);  view_215 = None
        permute_118: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_217: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_283, [2048, 2560])
        permute_119: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_63: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg174_1, view_217, permute_119);  arg174_1 = view_217 = permute_119 = None
        view_218: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [16, 128, 2560]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_221: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_218, [16, 128, -1, 80]);  view_218 = None
        permute_121: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_219: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_283, [2048, 2560]);  convert_element_type_283 = None
        permute_120: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg176_1, view_219, permute_120);  arg176_1 = view_219 = permute_120 = None
        view_220: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [16, 128, 2560]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_222: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_220, [16, 128, -1, 80]);  view_220 = None
        permute_122: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_25: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_25, full_default_24);  full_default_25 = full_default_24 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_118, permute_121, permute_122, where_19, False, scale = 0.11180339887498948);  permute_118 = permute_121 = permute_122 = where_19 = None
        getitem_87: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_123: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_87, [0, 2, 1, 3]);  getitem_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_223: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_123, [16, 128, -1]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_224: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_223, [2048, 2560]);  view_223 = None
        permute_124: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg178_1, view_224, permute_124);  arg178_1 = view_224 = permute_124 = None
        view_225: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [16, 128, 2560]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_84: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_81, view_225);  add_81 = view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_296: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_296, [2], correction = 0, keepdim = True)
        getitem_96: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_21[0]
        getitem_97: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_26: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_26 = None
        scalar_tensor_27: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_27 = None
        full_default_26: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_28: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_296, getitem_97);  convert_element_type_296 = getitem_97 = None
        add_85: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_21: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_79: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_21);  sub_28 = rsqrt_21 = None
        mul_80: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, arg179_1);  mul_79 = arg179_1 = None
        add_86: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_80, arg180_1);  mul_80 = arg180_1 = None
        convert_element_type_297: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_226: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_297, [2048, 2560]);  convert_element_type_297 = None
        permute_125: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg182_1, view_226, permute_125);  arg182_1 = view_226 = permute_125 = None
        view_227: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [16, 128, 2560]);  addmm_66 = None
        view_228: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [16, 128, -1, 80]);  view_227 = None
        permute_126: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_81: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_126, 0.334370152488211);  permute_126 = None
        expand_31: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_81, [16, 32, 128, 80]);  mul_81 = None
        clone_57: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_235: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [512, 128, 80]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_229: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_127: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg184_1, view_229, permute_127);  arg184_1 = view_229 = permute_127 = None
        view_230: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [16, 128, 2560]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_233: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_230, [16, 128, -1, 80]);  view_230 = None
        permute_129: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_131: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_129, [0, 1, 3, 2]);  permute_129 = None
        mul_82: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_131, 0.334370152488211);  permute_131 = None
        expand_32: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_82, [16, 32, 80, 128]);  mul_82 = None
        clone_58: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_236: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [512, 80, 128]);  clone_58 = None
        bmm_14: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_235, view_236);  view_235 = view_236 = None
        view_237: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [16, 32, 128, 128]);  bmm_14 = None
        eq_7: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_237, -inf)
        logical_not_14: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_default_27: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_309: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None
        amax_7: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_309, [-1], True)
        sub_29: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_309, amax_7);  convert_element_type_309 = amax_7 = None
        exp_7: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_8: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_310: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        where_21: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_15, full_default_27, convert_element_type_310);  logical_not_15 = full_default_27 = convert_element_type_310 = None
        expand_33: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_21, [16, 32, 128, 128]);  where_21 = None
        view_238: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_33, [512, 128, 128]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_231: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_128: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg186_1, view_231, permute_128);  arg186_1 = view_231 = permute_128 = None
        view_232: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [16, 128, 2560]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_234: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_232, [16, 128, -1, 80]);  view_232 = None
        permute_130: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_34: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_130, [16, 32, 128, 80]);  permute_130 = None
        clone_59: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_239: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [512, 128, 80]);  clone_59 = None
        bmm_15: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_238, view_239);  view_238 = view_239 = None
        view_240: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [16, 32, 128, 80]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_132: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
        clone_60: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_241: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [16, 128, -1]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_242: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_241, [2048, 2560]);  view_241 = None
        permute_133: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg188_1, view_242, permute_133);  arg188_1 = view_242 = permute_133 = None
        view_243: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [16, 128, 2560]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_88: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_84, view_243);  add_84 = view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_316: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_316, [2], correction = 0, keepdim = True)
        getitem_98: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_22[0]
        getitem_99: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_30: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_316, getitem_99);  convert_element_type_316 = getitem_99 = None
        add_89: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_22: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_83: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_22);  sub_30 = rsqrt_22 = None
        mul_84: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, arg189_1);  mul_83 = arg189_1 = None
        add_90: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_84, arg190_1);  mul_84 = arg190_1 = None
        convert_element_type_317: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_244: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_317, [2048, 2560]);  convert_element_type_317 = None
        permute_134: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, view_244, permute_134);  arg192_1 = view_244 = permute_134 = None
        view_245: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [16, 128, 10240]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_321: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_245, torch.float32);  view_245 = None
        mul_85: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, 0.5)
        mul_86: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, 0.7071067811865476);  convert_element_type_321 = None
        erf_7: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_91: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_87: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, add_91);  mul_85 = add_91 = None
        convert_element_type_322: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_246: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_322, [2048, 10240]);  convert_element_type_322 = None
        permute_135: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg194_1, view_246, permute_135);  arg194_1 = view_246 = permute_135 = None
        view_247: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [16, 128, 2560]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_92: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_88, view_247);  add_88 = view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_326: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_92, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_326, [2], correction = 0, keepdim = True)
        getitem_100: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_23[0]
        getitem_101: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_31: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_326, getitem_101);  convert_element_type_326 = getitem_101 = None
        add_93: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_23: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_88: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_23);  sub_31 = rsqrt_23 = None
        mul_89: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, arg195_1);  mul_88 = arg195_1 = None
        add_94: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_89, arg196_1);  mul_89 = arg196_1 = None
        convert_element_type_327: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_248: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_327, [2048, 2560])
        permute_136: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_72: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg198_1, view_248, permute_136);  arg198_1 = view_248 = permute_136 = None
        view_249: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [16, 128, 2560]);  addmm_72 = None
        view_250: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_249, [16, 128, -1, 80]);  view_249 = None
        permute_137: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_251: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_327, [2048, 2560])
        permute_138: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_73: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg200_1, view_251, permute_138);  arg200_1 = view_251 = permute_138 = None
        view_252: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [16, 128, 2560]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_255: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_252, [16, 128, -1, 80]);  view_252 = None
        permute_140: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_253: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_327, [2048, 2560]);  convert_element_type_327 = None
        permute_139: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_74: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg202_1, view_253, permute_139);  arg202_1 = view_253 = permute_139 = None
        view_254: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [16, 128, 2560]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_256: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_254, [16, 128, -1, 80]);  view_254 = None
        permute_141: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_29: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_29, full_default_28);  full_default_29 = full_default_28 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_137, permute_140, permute_141, where_22, False, scale = 0.11180339887498948);  permute_137 = permute_140 = permute_141 = where_22 = None
        getitem_102: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_142: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_102, [0, 2, 1, 3]);  getitem_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_257: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_142, [16, 128, -1]);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_258: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [2048, 2560]);  view_257 = None
        permute_143: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_75: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg204_1, view_258, permute_143);  arg204_1 = view_258 = permute_143 = None
        view_259: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [16, 128, 2560]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_95: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_92, view_259);  add_92 = view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_340: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_340, [2], correction = 0, keepdim = True)
        getitem_111: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_24[0]
        getitem_112: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_30: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_30 = None
        scalar_tensor_31: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_31 = None
        full_default_30: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_32: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_340, getitem_112);  convert_element_type_340 = getitem_112 = None
        add_96: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_111, 1e-05);  getitem_111 = None
        rsqrt_24: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_90: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_24);  sub_32 = rsqrt_24 = None
        mul_91: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, arg205_1);  mul_90 = arg205_1 = None
        add_97: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, arg206_1);  mul_91 = arg206_1 = None
        convert_element_type_341: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_260: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_341, [2048, 2560]);  convert_element_type_341 = None
        permute_144: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_76: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg208_1, view_260, permute_144);  arg208_1 = view_260 = permute_144 = None
        view_261: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [16, 128, 2560]);  addmm_76 = None
        view_262: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_261, [16, 128, -1, 80]);  view_261 = None
        permute_145: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_92: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_145, 0.334370152488211);  permute_145 = None
        expand_35: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_92, [16, 32, 128, 80]);  mul_92 = None
        clone_65: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_269: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [512, 128, 80]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_263: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_146: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_77: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg210_1, view_263, permute_146);  arg210_1 = view_263 = permute_146 = None
        view_264: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [16, 128, 2560]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_267: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_264, [16, 128, -1, 80]);  view_264 = None
        permute_148: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_150: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_148, [0, 1, 3, 2]);  permute_148 = None
        mul_93: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_150, 0.334370152488211);  permute_150 = None
        expand_36: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_93, [16, 32, 80, 128]);  mul_93 = None
        clone_66: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_270: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [512, 80, 128]);  clone_66 = None
        bmm_16: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_269, view_270);  view_269 = view_270 = None
        view_271: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [16, 32, 128, 128]);  bmm_16 = None
        eq_8: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_271, -inf)
        logical_not_16: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_default_31: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_353: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_271, torch.float32);  view_271 = None
        amax_8: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_353, [-1], True)
        sub_33: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_353, amax_8);  convert_element_type_353 = amax_8 = None
        exp_8: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_9: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_354: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        where_24: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_17, full_default_31, convert_element_type_354);  logical_not_17 = full_default_31 = convert_element_type_354 = None
        expand_37: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_24, [16, 32, 128, 128]);  where_24 = None
        view_272: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_37, [512, 128, 128]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_265: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_147: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_78: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg212_1, view_265, permute_147);  arg212_1 = view_265 = permute_147 = None
        view_266: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [16, 128, 2560]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_268: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_266, [16, 128, -1, 80]);  view_266 = None
        permute_149: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_38: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_149, [16, 32, 128, 80]);  permute_149 = None
        clone_67: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_273: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [512, 128, 80]);  clone_67 = None
        bmm_17: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_272, view_273);  view_272 = view_273 = None
        view_274: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [16, 32, 128, 80]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_151: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_68: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_275: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [16, 128, -1]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_276: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_275, [2048, 2560]);  view_275 = None
        permute_152: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_79: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg214_1, view_276, permute_152);  arg214_1 = view_276 = permute_152 = None
        view_277: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [16, 128, 2560]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_99: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_95, view_277);  add_95 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_360: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_360, [2], correction = 0, keepdim = True)
        getitem_113: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_25[0]
        getitem_114: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        sub_34: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_360, getitem_114);  convert_element_type_360 = getitem_114 = None
        add_100: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_113, 1e-05);  getitem_113 = None
        rsqrt_25: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_94: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_25);  sub_34 = rsqrt_25 = None
        mul_95: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, arg215_1);  mul_94 = arg215_1 = None
        add_101: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, arg216_1);  mul_95 = arg216_1 = None
        convert_element_type_361: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_278: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [2048, 2560]);  convert_element_type_361 = None
        permute_153: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_80: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg218_1, view_278, permute_153);  arg218_1 = view_278 = permute_153 = None
        view_279: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [16, 128, 10240]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_365: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        mul_96: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_365, 0.5)
        mul_97: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_365, 0.7071067811865476);  convert_element_type_365 = None
        erf_8: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_102: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_98: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, add_102);  mul_96 = add_102 = None
        convert_element_type_366: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_280: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_366, [2048, 10240]);  convert_element_type_366 = None
        permute_154: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_81: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg220_1, view_280, permute_154);  arg220_1 = view_280 = permute_154 = None
        view_281: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [16, 128, 2560]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_103: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_99, view_281);  add_99 = view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_370: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_370, [2], correction = 0, keepdim = True)
        getitem_115: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_26[0]
        getitem_116: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        sub_35: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_370, getitem_116);  convert_element_type_370 = getitem_116 = None
        add_104: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_26: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_99: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_26);  sub_35 = rsqrt_26 = None
        mul_100: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, arg221_1);  mul_99 = arg221_1 = None
        add_105: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_100, arg222_1);  mul_100 = arg222_1 = None
        convert_element_type_371: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_282: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_371, [2048, 2560])
        permute_155: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_82: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg224_1, view_282, permute_155);  arg224_1 = view_282 = permute_155 = None
        view_283: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [16, 128, 2560]);  addmm_82 = None
        view_284: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_283, [16, 128, -1, 80]);  view_283 = None
        permute_156: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_285: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_371, [2048, 2560])
        permute_157: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_83: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg226_1, view_285, permute_157);  arg226_1 = view_285 = permute_157 = None
        view_286: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [16, 128, 2560]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_289: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_286, [16, 128, -1, 80]);  view_286 = None
        permute_159: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_287: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_371, [2048, 2560]);  convert_element_type_371 = None
        permute_158: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_84: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg228_1, view_287, permute_158);  arg228_1 = view_287 = permute_158 = None
        view_288: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [16, 128, 2560]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_290: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_288, [16, 128, -1, 80]);  view_288 = None
        permute_160: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_33: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_33, full_default_32);  full_default_33 = full_default_32 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_156, permute_159, permute_160, where_25, False, scale = 0.11180339887498948);  permute_156 = permute_159 = permute_160 = where_25 = None
        getitem_117: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_291: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_161, [16, 128, -1]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_292: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_291, [2048, 2560]);  view_291 = None
        permute_162: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_85: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg230_1, view_292, permute_162);  arg230_1 = view_292 = permute_162 = None
        view_293: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_85, [16, 128, 2560]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_106: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_103, view_293);  add_103 = view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_384: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_384, [2], correction = 0, keepdim = True)
        getitem_126: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_27[0]
        getitem_127: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_34: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_34 = None
        scalar_tensor_35: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_35 = None
        full_default_34: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_36: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_384, getitem_127);  convert_element_type_384 = getitem_127 = None
        add_107: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_27: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_101: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_27);  sub_36 = rsqrt_27 = None
        mul_102: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, arg231_1);  mul_101 = arg231_1 = None
        add_108: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, arg232_1);  mul_102 = arg232_1 = None
        convert_element_type_385: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16);  add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_294: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_385, [2048, 2560]);  convert_element_type_385 = None
        permute_163: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_86: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg234_1, view_294, permute_163);  arg234_1 = view_294 = permute_163 = None
        view_295: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [16, 128, 2560]);  addmm_86 = None
        view_296: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [16, 128, -1, 80]);  view_295 = None
        permute_164: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_103: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_164, 0.334370152488211);  permute_164 = None
        expand_39: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_103, [16, 32, 128, 80]);  mul_103 = None
        clone_73: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_303: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [512, 128, 80]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_297: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_165: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_87: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg236_1, view_297, permute_165);  arg236_1 = view_297 = permute_165 = None
        view_298: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_87, [16, 128, 2560]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_301: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_298, [16, 128, -1, 80]);  view_298 = None
        permute_167: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_169: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_167, [0, 1, 3, 2]);  permute_167 = None
        mul_104: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_169, 0.334370152488211);  permute_169 = None
        expand_40: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_104, [16, 32, 80, 128]);  mul_104 = None
        clone_74: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_304: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [512, 80, 128]);  clone_74 = None
        bmm_18: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_303, view_304);  view_303 = view_304 = None
        view_305: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [16, 32, 128, 128]);  bmm_18 = None
        eq_9: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_305, -inf)
        logical_not_18: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_default_35: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_397: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        amax_9: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_397, [-1], True)
        sub_37: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_397, amax_9);  convert_element_type_397 = amax_9 = None
        exp_9: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_10: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_398: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        where_27: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_19, full_default_35, convert_element_type_398);  logical_not_19 = full_default_35 = convert_element_type_398 = None
        expand_41: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_27, [16, 32, 128, 128]);  where_27 = None
        view_306: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_41, [512, 128, 128]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_299: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_166: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_88: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg238_1, view_299, permute_166);  arg238_1 = view_299 = permute_166 = None
        view_300: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [16, 128, 2560]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_302: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_300, [16, 128, -1, 80]);  view_300 = None
        permute_168: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_42: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_168, [16, 32, 128, 80]);  permute_168 = None
        clone_75: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_307: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [512, 128, 80]);  clone_75 = None
        bmm_19: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_306, view_307);  view_306 = view_307 = None
        view_308: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [16, 32, 128, 80]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_170: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        clone_76: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_309: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [16, 128, -1]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_310: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [2048, 2560]);  view_309 = None
        permute_171: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_89: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg240_1, view_310, permute_171);  arg240_1 = view_310 = permute_171 = None
        view_311: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_89, [16, 128, 2560]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_110: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_106, view_311);  add_106 = view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_404: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_110, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_404, [2], correction = 0, keepdim = True)
        getitem_128: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_28[0]
        getitem_129: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        sub_38: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_404, getitem_129);  convert_element_type_404 = getitem_129 = None
        add_111: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_28: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_105: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_28);  sub_38 = rsqrt_28 = None
        mul_106: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, arg241_1);  mul_105 = arg241_1 = None
        add_112: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, arg242_1);  mul_106 = arg242_1 = None
        convert_element_type_405: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_312: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_405, [2048, 2560]);  convert_element_type_405 = None
        permute_172: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_90: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg244_1, view_312, permute_172);  arg244_1 = view_312 = permute_172 = None
        view_313: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [16, 128, 10240]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_409: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_313, torch.float32);  view_313 = None
        mul_107: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_409, 0.5)
        mul_108: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_409, 0.7071067811865476);  convert_element_type_409 = None
        erf_9: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
        add_113: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_109: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, add_113);  mul_107 = add_113 = None
        convert_element_type_410: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_314: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_410, [2048, 10240]);  convert_element_type_410 = None
        permute_173: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_91: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg246_1, view_314, permute_173);  arg246_1 = view_314 = permute_173 = None
        view_315: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_91, [16, 128, 2560]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_114: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, view_315);  add_110 = view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_414: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_414, [2], correction = 0, keepdim = True)
        getitem_130: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_29[0]
        getitem_131: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        sub_39: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_414, getitem_131);  convert_element_type_414 = getitem_131 = None
        add_115: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_29: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_110: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_29);  sub_39 = rsqrt_29 = None
        mul_111: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, arg247_1);  mul_110 = arg247_1 = None
        add_116: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, arg248_1);  mul_111 = arg248_1 = None
        convert_element_type_415: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_316: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_415, [2048, 2560])
        permute_174: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_92: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg250_1, view_316, permute_174);  arg250_1 = view_316 = permute_174 = None
        view_317: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_92, [16, 128, 2560]);  addmm_92 = None
        view_318: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_317, [16, 128, -1, 80]);  view_317 = None
        permute_175: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_319: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_415, [2048, 2560])
        permute_176: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_93: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg252_1, view_319, permute_176);  arg252_1 = view_319 = permute_176 = None
        view_320: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_93, [16, 128, 2560]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_323: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_320, [16, 128, -1, 80]);  view_320 = None
        permute_178: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_321: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_415, [2048, 2560]);  convert_element_type_415 = None
        permute_177: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_94: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg254_1, view_321, permute_177);  arg254_1 = view_321 = permute_177 = None
        view_322: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [16, 128, 2560]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_324: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_322, [16, 128, -1, 80]);  view_322 = None
        permute_179: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_37: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_37, full_default_36);  full_default_37 = full_default_36 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_175, permute_178, permute_179, where_28, False, scale = 0.11180339887498948);  permute_175 = permute_178 = permute_179 = where_28 = None
        getitem_132: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_180: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_325: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_180, [16, 128, -1]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_326: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_325, [2048, 2560]);  view_325 = None
        permute_181: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_95: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg256_1, view_326, permute_181);  arg256_1 = view_326 = permute_181 = None
        view_327: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_95, [16, 128, 2560]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_117: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_114, view_327);  add_114 = view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_428: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_428, [2], correction = 0, keepdim = True)
        getitem_141: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_30[0]
        getitem_142: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_38: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_38 = None
        scalar_tensor_39: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_39 = None
        full_default_38: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_40: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_428, getitem_142);  convert_element_type_428 = getitem_142 = None
        add_118: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_30: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_112: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_30);  sub_40 = rsqrt_30 = None
        mul_113: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, arg257_1);  mul_112 = arg257_1 = None
        add_119: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, arg258_1);  mul_113 = arg258_1 = None
        convert_element_type_429: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_328: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_429, [2048, 2560]);  convert_element_type_429 = None
        permute_182: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_96: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg260_1, view_328, permute_182);  arg260_1 = view_328 = permute_182 = None
        view_329: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_96, [16, 128, 2560]);  addmm_96 = None
        view_330: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [16, 128, -1, 80]);  view_329 = None
        permute_183: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_114: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_183, 0.334370152488211);  permute_183 = None
        expand_43: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_114, [16, 32, 128, 80]);  mul_114 = None
        clone_81: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_337: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [512, 128, 80]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_331: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_184: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_97: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg262_1, view_331, permute_184);  arg262_1 = view_331 = permute_184 = None
        view_332: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_97, [16, 128, 2560]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_335: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_332, [16, 128, -1, 80]);  view_332 = None
        permute_186: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_188: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_186, [0, 1, 3, 2]);  permute_186 = None
        mul_115: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_188, 0.334370152488211);  permute_188 = None
        expand_44: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_115, [16, 32, 80, 128]);  mul_115 = None
        clone_82: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_338: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [512, 80, 128]);  clone_82 = None
        bmm_20: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_337, view_338);  view_337 = view_338 = None
        view_339: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [16, 32, 128, 128]);  bmm_20 = None
        eq_10: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_339, -inf)
        logical_not_20: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_default_39: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_441: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        amax_10: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_441, [-1], True)
        sub_41: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_441, amax_10);  convert_element_type_441 = amax_10 = None
        exp_10: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_11: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_442: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        where_30: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_21, full_default_39, convert_element_type_442);  logical_not_21 = full_default_39 = convert_element_type_442 = None
        expand_45: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_30, [16, 32, 128, 128]);  where_30 = None
        view_340: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_45, [512, 128, 128]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_333: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_185: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_98: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg264_1, view_333, permute_185);  arg264_1 = view_333 = permute_185 = None
        view_334: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_98, [16, 128, 2560]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_336: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_334, [16, 128, -1, 80]);  view_334 = None
        permute_187: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_46: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_187, [16, 32, 128, 80]);  permute_187 = None
        clone_83: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_341: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [512, 128, 80]);  clone_83 = None
        bmm_21: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_340, view_341);  view_340 = view_341 = None
        view_342: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [16, 32, 128, 80]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_189: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
        clone_84: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_343: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [16, 128, -1]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_344: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_343, [2048, 2560]);  view_343 = None
        permute_190: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_99: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg266_1, view_344, permute_190);  arg266_1 = view_344 = permute_190 = None
        view_345: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_99, [16, 128, 2560]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_121: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, view_345);  add_117 = view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_448: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_448, [2], correction = 0, keepdim = True)
        getitem_143: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_31[0]
        getitem_144: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        sub_42: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_448, getitem_144);  convert_element_type_448 = getitem_144 = None
        add_122: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_31: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_116: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_31);  sub_42 = rsqrt_31 = None
        mul_117: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, arg267_1);  mul_116 = arg267_1 = None
        add_123: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_117, arg268_1);  mul_117 = arg268_1 = None
        convert_element_type_449: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_346: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_449, [2048, 2560]);  convert_element_type_449 = None
        permute_191: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_100: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg270_1, view_346, permute_191);  arg270_1 = view_346 = permute_191 = None
        view_347: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_100, [16, 128, 10240]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_453: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_347, torch.float32);  view_347 = None
        mul_118: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_453, 0.5)
        mul_119: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_453, 0.7071067811865476);  convert_element_type_453 = None
        erf_10: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_124: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_120: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, add_124);  mul_118 = add_124 = None
        convert_element_type_454: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_348: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_454, [2048, 10240]);  convert_element_type_454 = None
        permute_192: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_101: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg272_1, view_348, permute_192);  arg272_1 = view_348 = permute_192 = None
        view_349: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_101, [16, 128, 2560]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_125: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, view_349);  add_121 = view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_458: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_458, [2], correction = 0, keepdim = True)
        getitem_145: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_32[0]
        getitem_146: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        sub_43: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_458, getitem_146);  convert_element_type_458 = getitem_146 = None
        add_126: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_145, 1e-05);  getitem_145 = None
        rsqrt_32: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_121: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_32);  sub_43 = rsqrt_32 = None
        mul_122: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, arg273_1);  mul_121 = arg273_1 = None
        add_127: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, arg274_1);  mul_122 = arg274_1 = None
        convert_element_type_459: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_350: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_459, [2048, 2560])
        permute_193: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_102: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg276_1, view_350, permute_193);  arg276_1 = view_350 = permute_193 = None
        view_351: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_102, [16, 128, 2560]);  addmm_102 = None
        view_352: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_351, [16, 128, -1, 80]);  view_351 = None
        permute_194: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_353: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_459, [2048, 2560])
        permute_195: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_103: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg278_1, view_353, permute_195);  arg278_1 = view_353 = permute_195 = None
        view_354: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_103, [16, 128, 2560]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_357: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_354, [16, 128, -1, 80]);  view_354 = None
        permute_197: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_355: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_459, [2048, 2560]);  convert_element_type_459 = None
        permute_196: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_104: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg280_1, view_355, permute_196);  arg280_1 = view_355 = permute_196 = None
        view_356: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_104, [16, 128, 2560]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_358: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_356, [16, 128, -1, 80]);  view_356 = None
        permute_198: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_41: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_41, full_default_40);  full_default_41 = full_default_40 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_194, permute_197, permute_198, where_31, False, scale = 0.11180339887498948);  permute_194 = permute_197 = permute_198 = where_31 = None
        getitem_147: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_199: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1, 3]);  getitem_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_359: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_199, [16, 128, -1]);  permute_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_360: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_359, [2048, 2560]);  view_359 = None
        permute_200: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_105: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg282_1, view_360, permute_200);  arg282_1 = view_360 = permute_200 = None
        view_361: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_105, [16, 128, 2560]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_128: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, view_361);  add_125 = view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_472: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_128, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_472, [2], correction = 0, keepdim = True)
        getitem_156: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_33[0]
        getitem_157: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_42: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_42 = None
        scalar_tensor_43: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_43 = None
        full_default_42: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_44: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_472, getitem_157);  convert_element_type_472 = getitem_157 = None
        add_129: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_33: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_123: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_33);  sub_44 = rsqrt_33 = None
        mul_124: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, arg283_1);  mul_123 = arg283_1 = None
        add_130: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_124, arg284_1);  mul_124 = arg284_1 = None
        convert_element_type_473: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16);  add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_362: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_473, [2048, 2560]);  convert_element_type_473 = None
        permute_201: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_106: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg286_1, view_362, permute_201);  arg286_1 = view_362 = permute_201 = None
        view_363: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_106, [16, 128, 2560]);  addmm_106 = None
        view_364: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_363, [16, 128, -1, 80]);  view_363 = None
        permute_202: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_125: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_202, 0.334370152488211);  permute_202 = None
        expand_47: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_125, [16, 32, 128, 80]);  mul_125 = None
        clone_89: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_371: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [512, 128, 80]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_365: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_203: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_107: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg288_1, view_365, permute_203);  arg288_1 = view_365 = permute_203 = None
        view_366: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_107, [16, 128, 2560]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_369: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_366, [16, 128, -1, 80]);  view_366 = None
        permute_205: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_207: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_205, [0, 1, 3, 2]);  permute_205 = None
        mul_126: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_207, 0.334370152488211);  permute_207 = None
        expand_48: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_126, [16, 32, 80, 128]);  mul_126 = None
        clone_90: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_372: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [512, 80, 128]);  clone_90 = None
        bmm_22: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_371, view_372);  view_371 = view_372 = None
        view_373: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [16, 32, 128, 128]);  bmm_22 = None
        eq_11: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_373, -inf)
        logical_not_22: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_default_43: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_485: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_373, torch.float32);  view_373 = None
        amax_11: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_485, [-1], True)
        sub_45: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_485, amax_11);  convert_element_type_485 = amax_11 = None
        exp_11: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_12: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_486: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        where_33: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_23, full_default_43, convert_element_type_486);  logical_not_23 = full_default_43 = convert_element_type_486 = None
        expand_49: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_33, [16, 32, 128, 128]);  where_33 = None
        view_374: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_49, [512, 128, 128]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_367: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_204: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_108: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg290_1, view_367, permute_204);  arg290_1 = view_367 = permute_204 = None
        view_368: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_108, [16, 128, 2560]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_370: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_368, [16, 128, -1, 80]);  view_368 = None
        permute_206: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_50: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_206, [16, 32, 128, 80]);  permute_206 = None
        clone_91: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_375: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [512, 128, 80]);  clone_91 = None
        bmm_23: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_374, view_375);  view_374 = view_375 = None
        view_376: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [16, 32, 128, 80]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_208: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        clone_92: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_377: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [16, 128, -1]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_378: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_377, [2048, 2560]);  view_377 = None
        permute_209: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_109: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg292_1, view_378, permute_209);  arg292_1 = view_378 = permute_209 = None
        view_379: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_109, [16, 128, 2560]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_132: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_128, view_379);  add_128 = view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_492: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_492, [2], correction = 0, keepdim = True)
        getitem_158: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_34[0]
        getitem_159: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        sub_46: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_492, getitem_159);  convert_element_type_492 = getitem_159 = None
        add_133: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_34: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_127: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_34);  sub_46 = rsqrt_34 = None
        mul_128: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, arg293_1);  mul_127 = arg293_1 = None
        add_134: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_128, arg294_1);  mul_128 = arg294_1 = None
        convert_element_type_493: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_380: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_493, [2048, 2560]);  convert_element_type_493 = None
        permute_210: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_110: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg296_1, view_380, permute_210);  arg296_1 = view_380 = permute_210 = None
        view_381: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_110, [16, 128, 10240]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_497: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_381, torch.float32);  view_381 = None
        mul_129: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.5)
        mul_130: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.7071067811865476);  convert_element_type_497 = None
        erf_11: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
        add_135: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_131: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, add_135);  mul_129 = add_135 = None
        convert_element_type_498: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_131, torch.bfloat16);  mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_382: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_498, [2048, 10240]);  convert_element_type_498 = None
        permute_211: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_111: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg298_1, view_382, permute_211);  arg298_1 = view_382 = permute_211 = None
        view_383: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_111, [16, 128, 2560]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_136: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_132, view_383);  add_132 = view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_502: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_502, [2], correction = 0, keepdim = True)
        getitem_160: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_35[0]
        getitem_161: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        sub_47: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_502, getitem_161);  convert_element_type_502 = getitem_161 = None
        add_137: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_35: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        mul_132: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_35);  sub_47 = rsqrt_35 = None
        mul_133: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, arg299_1);  mul_132 = arg299_1 = None
        add_138: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_133, arg300_1);  mul_133 = arg300_1 = None
        convert_element_type_503: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16);  add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_384: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_503, [2048, 2560])
        permute_212: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_112: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg302_1, view_384, permute_212);  arg302_1 = view_384 = permute_212 = None
        view_385: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_112, [16, 128, 2560]);  addmm_112 = None
        view_386: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_385, [16, 128, -1, 80]);  view_385 = None
        permute_213: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_387: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_503, [2048, 2560])
        permute_214: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_113: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg304_1, view_387, permute_214);  arg304_1 = view_387 = permute_214 = None
        view_388: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_113, [16, 128, 2560]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_391: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_388, [16, 128, -1, 80]);  view_388 = None
        permute_216: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_389: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_503, [2048, 2560]);  convert_element_type_503 = None
        permute_215: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_114: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg306_1, view_389, permute_215);  arg306_1 = view_389 = permute_215 = None
        view_390: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_114, [16, 128, 2560]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_392: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_390, [16, 128, -1, 80]);  view_390 = None
        permute_217: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_45: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_45, full_default_44);  full_default_45 = full_default_44 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_213, permute_216, permute_217, where_34, False, scale = 0.11180339887498948);  permute_213 = permute_216 = permute_217 = where_34 = None
        getitem_162: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_218: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_393: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_218, [16, 128, -1]);  permute_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_394: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_393, [2048, 2560]);  view_393 = None
        permute_219: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_115: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg308_1, view_394, permute_219);  arg308_1 = view_394 = permute_219 = None
        view_395: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_115, [16, 128, 2560]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_139: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_136, view_395);  add_136 = view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_516: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_516, [2], correction = 0, keepdim = True)
        getitem_171: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_36[0]
        getitem_172: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_46: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_46 = None
        scalar_tensor_47: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_47 = None
        full_default_46: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_48: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_516, getitem_172);  convert_element_type_516 = getitem_172 = None
        add_140: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_171, 1e-05);  getitem_171 = None
        rsqrt_36: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_134: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_36);  sub_48 = rsqrt_36 = None
        mul_135: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, arg309_1);  mul_134 = arg309_1 = None
        add_141: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_135, arg310_1);  mul_135 = arg310_1 = None
        convert_element_type_517: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_396: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_517, [2048, 2560]);  convert_element_type_517 = None
        permute_220: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_116: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg312_1, view_396, permute_220);  arg312_1 = view_396 = permute_220 = None
        view_397: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_116, [16, 128, 2560]);  addmm_116 = None
        view_398: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [16, 128, -1, 80]);  view_397 = None
        permute_221: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_136: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_221, 0.334370152488211);  permute_221 = None
        expand_51: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_136, [16, 32, 128, 80]);  mul_136 = None
        clone_97: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_405: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [512, 128, 80]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_399: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_222: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_117: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg314_1, view_399, permute_222);  arg314_1 = view_399 = permute_222 = None
        view_400: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_117, [16, 128, 2560]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_403: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_400, [16, 128, -1, 80]);  view_400 = None
        permute_224: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_226: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_224, [0, 1, 3, 2]);  permute_224 = None
        mul_137: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_226, 0.334370152488211);  permute_226 = None
        expand_52: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_137, [16, 32, 80, 128]);  mul_137 = None
        clone_98: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_406: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [512, 80, 128]);  clone_98 = None
        bmm_24: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_405, view_406);  view_405 = view_406 = None
        view_407: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [16, 32, 128, 128]);  bmm_24 = None
        eq_12: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_407, -inf)
        logical_not_24: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_12);  eq_12 = None
        any_13: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_24, -1, True);  logical_not_24 = None
        logical_not_25: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        full_default_47: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_529: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        amax_12: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_529, [-1], True)
        sub_49: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_529, amax_12);  convert_element_type_529 = amax_12 = None
        exp_12: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_13: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_530: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        where_36: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_25, full_default_47, convert_element_type_530);  logical_not_25 = full_default_47 = convert_element_type_530 = None
        expand_53: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_36, [16, 32, 128, 128]);  where_36 = None
        view_408: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_53, [512, 128, 128]);  expand_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_401: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_223: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_118: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg316_1, view_401, permute_223);  arg316_1 = view_401 = permute_223 = None
        view_402: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_118, [16, 128, 2560]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_404: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_402, [16, 128, -1, 80]);  view_402 = None
        permute_225: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_54: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_225, [16, 32, 128, 80]);  permute_225 = None
        clone_99: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_409: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [512, 128, 80]);  clone_99 = None
        bmm_25: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = view_409 = None
        view_410: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [16, 32, 128, 80]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_100: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_411: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [16, 128, -1]);  clone_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_412: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [2048, 2560]);  view_411 = None
        permute_228: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_119: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg318_1, view_412, permute_228);  arg318_1 = view_412 = permute_228 = None
        view_413: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_119, [16, 128, 2560]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_143: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_139, view_413);  add_139 = view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_536: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_536, [2], correction = 0, keepdim = True)
        getitem_173: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_37[0]
        getitem_174: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        sub_50: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_536, getitem_174);  convert_element_type_536 = getitem_174 = None
        add_144: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_173, 1e-05);  getitem_173 = None
        rsqrt_37: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_138: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_37);  sub_50 = rsqrt_37 = None
        mul_139: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, arg319_1);  mul_138 = arg319_1 = None
        add_145: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, arg320_1);  mul_139 = arg320_1 = None
        convert_element_type_537: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_414: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_537, [2048, 2560]);  convert_element_type_537 = None
        permute_229: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_120: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg322_1, view_414, permute_229);  arg322_1 = view_414 = permute_229 = None
        view_415: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_120, [16, 128, 10240]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_541: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_140: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_541, 0.5)
        mul_141: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_541, 0.7071067811865476);  convert_element_type_541 = None
        erf_12: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_146: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_142: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, add_146);  mul_140 = add_146 = None
        convert_element_type_542: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_416: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_542, [2048, 10240]);  convert_element_type_542 = None
        permute_230: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_121: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg324_1, view_416, permute_230);  arg324_1 = view_416 = permute_230 = None
        view_417: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_121, [16, 128, 2560]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_147: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_143, view_417);  add_143 = view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_546: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_546, [2], correction = 0, keepdim = True)
        getitem_175: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_38[0]
        getitem_176: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        sub_51: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_546, getitem_176);  convert_element_type_546 = getitem_176 = None
        add_148: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_175, 1e-05);  getitem_175 = None
        rsqrt_38: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_143: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_38);  sub_51 = rsqrt_38 = None
        mul_144: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, arg325_1);  mul_143 = arg325_1 = None
        add_149: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, arg326_1);  mul_144 = arg326_1 = None
        convert_element_type_547: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_418: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_547, [2048, 2560])
        permute_231: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_122: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg328_1, view_418, permute_231);  arg328_1 = view_418 = permute_231 = None
        view_419: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_122, [16, 128, 2560]);  addmm_122 = None
        view_420: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_419, [16, 128, -1, 80]);  view_419 = None
        permute_232: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_421: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_547, [2048, 2560])
        permute_233: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_123: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg330_1, view_421, permute_233);  arg330_1 = view_421 = permute_233 = None
        view_422: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_123, [16, 128, 2560]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_425: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_422, [16, 128, -1, 80]);  view_422 = None
        permute_235: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_423: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_547, [2048, 2560]);  convert_element_type_547 = None
        permute_234: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_124: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg332_1, view_423, permute_234);  arg332_1 = view_423 = permute_234 = None
        view_424: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_124, [16, 128, 2560]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_426: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_424, [16, 128, -1, 80]);  view_424 = None
        permute_236: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_49: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_37: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_49, full_default_48);  full_default_49 = full_default_48 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_232, permute_235, permute_236, where_37, False, scale = 0.11180339887498948);  permute_232 = permute_235 = permute_236 = where_37 = None
        getitem_177: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_237: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_177, [0, 2, 1, 3]);  getitem_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_427: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_237, [16, 128, -1]);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_428: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_427, [2048, 2560]);  view_427 = None
        permute_238: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_125: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg334_1, view_428, permute_238);  arg334_1 = view_428 = permute_238 = None
        view_429: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_125, [16, 128, 2560]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_150: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_147, view_429);  add_147 = view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_560: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_150, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_560, [2], correction = 0, keepdim = True)
        getitem_186: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_39[0]
        getitem_187: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_50: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_50 = None
        scalar_tensor_51: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_51 = None
        full_default_50: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_52: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_560, getitem_187);  convert_element_type_560 = getitem_187 = None
        add_151: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_39: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_145: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_39);  sub_52 = rsqrt_39 = None
        mul_146: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, arg335_1);  mul_145 = arg335_1 = None
        add_152: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, arg336_1);  mul_146 = arg336_1 = None
        convert_element_type_561: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_430: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_561, [2048, 2560]);  convert_element_type_561 = None
        permute_239: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_126: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg338_1, view_430, permute_239);  arg338_1 = view_430 = permute_239 = None
        view_431: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_126, [16, 128, 2560]);  addmm_126 = None
        view_432: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_431, [16, 128, -1, 80]);  view_431 = None
        permute_240: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_147: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_240, 0.334370152488211);  permute_240 = None
        expand_55: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_147, [16, 32, 128, 80]);  mul_147 = None
        clone_105: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_439: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [512, 128, 80]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_433: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_241: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_127: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg340_1, view_433, permute_241);  arg340_1 = view_433 = permute_241 = None
        view_434: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_127, [16, 128, 2560]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_437: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_434, [16, 128, -1, 80]);  view_434 = None
        permute_243: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_245: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_243, [0, 1, 3, 2]);  permute_243 = None
        mul_148: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_245, 0.334370152488211);  permute_245 = None
        expand_56: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_148, [16, 32, 80, 128]);  mul_148 = None
        clone_106: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_440: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [512, 80, 128]);  clone_106 = None
        bmm_26: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_439, view_440);  view_439 = view_440 = None
        view_441: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [16, 32, 128, 128]);  bmm_26 = None
        eq_13: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_441, -inf)
        logical_not_26: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_13);  eq_13 = None
        any_14: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_26, -1, True);  logical_not_26 = None
        logical_not_27: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        full_default_51: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_573: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_441, torch.float32);  view_441 = None
        amax_13: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_573, [-1], True)
        sub_53: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_573, amax_13);  convert_element_type_573 = amax_13 = None
        exp_13: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
        sum_14: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_574: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        where_39: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_27, full_default_51, convert_element_type_574);  logical_not_27 = full_default_51 = convert_element_type_574 = None
        expand_57: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_39, [16, 32, 128, 128]);  where_39 = None
        view_442: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_57, [512, 128, 128]);  expand_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_435: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_242: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_128: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg342_1, view_435, permute_242);  arg342_1 = view_435 = permute_242 = None
        view_436: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_128, [16, 128, 2560]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_438: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_436, [16, 128, -1, 80]);  view_436 = None
        permute_244: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_438, [0, 2, 1, 3]);  view_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_58: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_244, [16, 32, 128, 80]);  permute_244 = None
        clone_107: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_443: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [512, 128, 80]);  clone_107 = None
        bmm_27: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_442, view_443);  view_442 = view_443 = None
        view_444: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [16, 32, 128, 80]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_246: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
        clone_108: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_445: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [16, 128, -1]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_446: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_445, [2048, 2560]);  view_445 = None
        permute_247: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_129: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg344_1, view_446, permute_247);  arg344_1 = view_446 = permute_247 = None
        view_447: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_129, [16, 128, 2560]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_154: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_150, view_447);  add_150 = view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_580: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_580, [2], correction = 0, keepdim = True)
        getitem_188: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_40[0]
        getitem_189: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        sub_54: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_580, getitem_189);  convert_element_type_580 = getitem_189 = None
        add_155: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_40: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_149: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_40);  sub_54 = rsqrt_40 = None
        mul_150: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_149, arg345_1);  mul_149 = arg345_1 = None
        add_156: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_150, arg346_1);  mul_150 = arg346_1 = None
        convert_element_type_581: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_156, torch.bfloat16);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_448: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_581, [2048, 2560]);  convert_element_type_581 = None
        permute_248: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_130: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg348_1, view_448, permute_248);  arg348_1 = view_448 = permute_248 = None
        view_449: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_130, [16, 128, 10240]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_585: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_449, torch.float32);  view_449 = None
        mul_151: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_585, 0.5)
        mul_152: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_585, 0.7071067811865476);  convert_element_type_585 = None
        erf_13: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
        add_157: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_153: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, add_157);  mul_151 = add_157 = None
        convert_element_type_586: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_153, torch.bfloat16);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_450: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_586, [2048, 10240]);  convert_element_type_586 = None
        permute_249: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_131: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg350_1, view_450, permute_249);  arg350_1 = view_450 = permute_249 = None
        view_451: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_131, [16, 128, 2560]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_158: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_154, view_451);  add_154 = view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_590: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_590, [2], correction = 0, keepdim = True)
        getitem_190: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_41[0]
        getitem_191: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        sub_55: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_590, getitem_191);  convert_element_type_590 = getitem_191 = None
        add_159: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_41: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_154: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_41);  sub_55 = rsqrt_41 = None
        mul_155: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, arg351_1);  mul_154 = arg351_1 = None
        add_160: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, arg352_1);  mul_155 = arg352_1 = None
        convert_element_type_591: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_452: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_591, [2048, 2560])
        permute_250: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_132: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg354_1, view_452, permute_250);  arg354_1 = view_452 = permute_250 = None
        view_453: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_132, [16, 128, 2560]);  addmm_132 = None
        view_454: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_453, [16, 128, -1, 80]);  view_453 = None
        permute_251: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_455: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_591, [2048, 2560])
        permute_252: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        addmm_133: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg356_1, view_455, permute_252);  arg356_1 = view_455 = permute_252 = None
        view_456: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_133, [16, 128, 2560]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_459: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_456, [16, 128, -1, 80]);  view_456 = None
        permute_254: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_457: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_591, [2048, 2560]);  convert_element_type_591 = None
        permute_253: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_134: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg358_1, view_457, permute_253);  arg358_1 = view_457 = permute_253 = None
        view_458: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_134, [16, 128, 2560]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_460: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_458, [16, 128, -1, 80]);  view_458 = None
        permute_255: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_53: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_53, full_default_52);  full_default_53 = full_default_52 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_251, permute_254, permute_255, where_40, False, scale = 0.11180339887498948);  permute_251 = permute_254 = permute_255 = where_40 = None
        getitem_192: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_256: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_192, [0, 2, 1, 3]);  getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_461: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_256, [16, 128, -1]);  permute_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_462: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_461, [2048, 2560]);  view_461 = None
        permute_257: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_135: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg360_1, view_462, permute_257);  arg360_1 = view_462 = permute_257 = None
        view_463: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_135, [16, 128, 2560]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_161: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_158, view_463);  add_158 = view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_604: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_604, [2], correction = 0, keepdim = True)
        getitem_201: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_42[0]
        getitem_202: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_54: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_54 = None
        scalar_tensor_55: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_55 = None
        full_default_54: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_56: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_604, getitem_202);  convert_element_type_604 = getitem_202 = None
        add_162: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_201, 1e-05);  getitem_201 = None
        rsqrt_42: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_156: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_42);  sub_56 = rsqrt_42 = None
        mul_157: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, arg361_1);  mul_156 = arg361_1 = None
        add_163: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_157, arg362_1);  mul_157 = arg362_1 = None
        convert_element_type_605: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_163, torch.bfloat16);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_464: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_605, [2048, 2560]);  convert_element_type_605 = None
        permute_258: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_136: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg364_1, view_464, permute_258);  arg364_1 = view_464 = permute_258 = None
        view_465: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_136, [16, 128, 2560]);  addmm_136 = None
        view_466: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_465, [16, 128, -1, 80]);  view_465 = None
        permute_259: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_158: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_259, 0.334370152488211);  permute_259 = None
        expand_59: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_158, [16, 32, 128, 80]);  mul_158 = None
        clone_113: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_473: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [512, 128, 80]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_467: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_260: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg365_1, [1, 0]);  arg365_1 = None
        addmm_137: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg366_1, view_467, permute_260);  arg366_1 = view_467 = permute_260 = None
        view_468: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_137, [16, 128, 2560]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_471: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_468, [16, 128, -1, 80]);  view_468 = None
        permute_262: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_264: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_262, [0, 1, 3, 2]);  permute_262 = None
        mul_159: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_264, 0.334370152488211);  permute_264 = None
        expand_60: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_159, [16, 32, 80, 128]);  mul_159 = None
        clone_114: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_474: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [512, 80, 128]);  clone_114 = None
        bmm_28: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_473, view_474);  view_473 = view_474 = None
        view_475: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [16, 32, 128, 128]);  bmm_28 = None
        eq_14: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_475, -inf)
        logical_not_28: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_14);  eq_14 = None
        any_15: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_28, -1, True);  logical_not_28 = None
        logical_not_29: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        full_default_55: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_617: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_475, torch.float32);  view_475 = None
        amax_14: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_617, [-1], True)
        sub_57: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_617, amax_14);  convert_element_type_617 = amax_14 = None
        exp_14: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_15: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_618: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        where_42: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_29, full_default_55, convert_element_type_618);  logical_not_29 = full_default_55 = convert_element_type_618 = None
        expand_61: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_42, [16, 32, 128, 128]);  where_42 = None
        view_476: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_61, [512, 128, 128]);  expand_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_469: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_261: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_138: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg368_1, view_469, permute_261);  arg368_1 = view_469 = permute_261 = None
        view_470: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_138, [16, 128, 2560]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_472: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_470, [16, 128, -1, 80]);  view_470 = None
        permute_263: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_62: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_263, [16, 32, 128, 80]);  permute_263 = None
        clone_115: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_477: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [512, 128, 80]);  clone_115 = None
        bmm_29: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_476, view_477);  view_476 = view_477 = None
        view_478: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [16, 32, 128, 80]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_265: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
        clone_116: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_479: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [16, 128, -1]);  clone_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_480: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_479, [2048, 2560]);  view_479 = None
        permute_266: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_139: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg370_1, view_480, permute_266);  arg370_1 = view_480 = permute_266 = None
        view_481: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_139, [16, 128, 2560]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_165: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_161, view_481);  add_161 = view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_624: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_165, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_624, [2], correction = 0, keepdim = True)
        getitem_203: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_43[0]
        getitem_204: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        sub_58: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_624, getitem_204);  convert_element_type_624 = getitem_204 = None
        add_166: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_203, 1e-05);  getitem_203 = None
        rsqrt_43: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_160: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_43);  sub_58 = rsqrt_43 = None
        mul_161: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, arg371_1);  mul_160 = arg371_1 = None
        add_167: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, arg372_1);  mul_161 = arg372_1 = None
        convert_element_type_625: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.bfloat16);  add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_482: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_625, [2048, 2560]);  convert_element_type_625 = None
        permute_267: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_140: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg374_1, view_482, permute_267);  arg374_1 = view_482 = permute_267 = None
        view_483: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_140, [16, 128, 10240]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_629: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_483, torch.float32);  view_483 = None
        mul_162: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_629, 0.5)
        mul_163: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_629, 0.7071067811865476);  convert_element_type_629 = None
        erf_14: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_168: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_164: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, add_168);  mul_162 = add_168 = None
        convert_element_type_630: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_164, torch.bfloat16);  mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_484: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_630, [2048, 10240]);  convert_element_type_630 = None
        permute_268: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        addmm_141: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg376_1, view_484, permute_268);  arg376_1 = view_484 = permute_268 = None
        view_485: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_141, [16, 128, 2560]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_169: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_165, view_485);  add_165 = view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_634: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_634, [2], correction = 0, keepdim = True)
        getitem_205: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_44[0]
        getitem_206: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        sub_59: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_634, getitem_206);  convert_element_type_634 = getitem_206 = None
        add_170: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_205, 1e-05);  getitem_205 = None
        rsqrt_44: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_165: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_44);  sub_59 = rsqrt_44 = None
        mul_166: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, arg377_1);  mul_165 = arg377_1 = None
        add_171: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_166, arg378_1);  mul_166 = arg378_1 = None
        convert_element_type_635: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16);  add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_486: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_635, [2048, 2560])
        permute_269: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_142: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg380_1, view_486, permute_269);  arg380_1 = view_486 = permute_269 = None
        view_487: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_142, [16, 128, 2560]);  addmm_142 = None
        view_488: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_487, [16, 128, -1, 80]);  view_487 = None
        permute_270: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_489: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_635, [2048, 2560])
        permute_271: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
        addmm_143: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg382_1, view_489, permute_271);  arg382_1 = view_489 = permute_271 = None
        view_490: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_143, [16, 128, 2560]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_493: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_490, [16, 128, -1, 80]);  view_490 = None
        permute_273: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_491: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_635, [2048, 2560]);  convert_element_type_635 = None
        permute_272: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_144: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg384_1, view_491, permute_272);  arg384_1 = view_491 = permute_272 = None
        view_492: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_144, [16, 128, 2560]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_494: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_492, [16, 128, -1, 80]);  view_492 = None
        permute_274: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_57: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_43: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_57, full_default_56);  full_default_57 = full_default_56 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_270, permute_273, permute_274, where_43, False, scale = 0.11180339887498948);  permute_270 = permute_273 = permute_274 = where_43 = None
        getitem_207: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_275: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_495: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_275, [16, 128, -1]);  permute_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_496: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_495, [2048, 2560]);  view_495 = None
        permute_276: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_145: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg386_1, view_496, permute_276);  arg386_1 = view_496 = permute_276 = None
        view_497: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_145, [16, 128, 2560]);  addmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_172: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_169, view_497);  add_169 = view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_648: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_172, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_648, [2], correction = 0, keepdim = True)
        getitem_216: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_45[0]
        getitem_217: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_58: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_58 = None
        scalar_tensor_59: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_59 = None
        full_default_58: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_60: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_648, getitem_217);  convert_element_type_648 = getitem_217 = None
        add_173: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_45: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        mul_167: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_45);  sub_60 = rsqrt_45 = None
        mul_168: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, arg387_1);  mul_167 = arg387_1 = None
        add_174: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_168, arg388_1);  mul_168 = arg388_1 = None
        convert_element_type_649: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.bfloat16);  add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_498: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_649, [2048, 2560]);  convert_element_type_649 = None
        permute_277: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        addmm_146: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg390_1, view_498, permute_277);  arg390_1 = view_498 = permute_277 = None
        view_499: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_146, [16, 128, 2560]);  addmm_146 = None
        view_500: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_499, [16, 128, -1, 80]);  view_499 = None
        permute_278: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_169: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_278, 0.334370152488211);  permute_278 = None
        expand_63: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_169, [16, 32, 128, 80]);  mul_169 = None
        clone_121: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_507: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [512, 128, 80]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_501: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_279: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
        addmm_147: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg392_1, view_501, permute_279);  arg392_1 = view_501 = permute_279 = None
        view_502: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_147, [16, 128, 2560]);  addmm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_505: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_502, [16, 128, -1, 80]);  view_502 = None
        permute_281: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_283: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_281, [0, 1, 3, 2]);  permute_281 = None
        mul_170: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_283, 0.334370152488211);  permute_283 = None
        expand_64: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_170, [16, 32, 80, 128]);  mul_170 = None
        clone_122: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_508: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [512, 80, 128]);  clone_122 = None
        bmm_30: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_507, view_508);  view_507 = view_508 = None
        view_509: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [16, 32, 128, 128]);  bmm_30 = None
        eq_15: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_509, -inf)
        logical_not_30: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_15);  eq_15 = None
        any_16: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_30, -1, True);  logical_not_30 = None
        logical_not_31: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        full_default_59: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_661: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_509, torch.float32);  view_509 = None
        amax_15: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_661, [-1], True)
        sub_61: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_661, amax_15);  convert_element_type_661 = amax_15 = None
        exp_15: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_16: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_662: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        where_45: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_31, full_default_59, convert_element_type_662);  logical_not_31 = full_default_59 = convert_element_type_662 = None
        expand_65: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_45, [16, 32, 128, 128]);  where_45 = None
        view_510: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_65, [512, 128, 128]);  expand_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_503: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_280: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
        addmm_148: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg394_1, view_503, permute_280);  arg394_1 = view_503 = permute_280 = None
        view_504: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_148, [16, 128, 2560]);  addmm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_506: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_504, [16, 128, -1, 80]);  view_504 = None
        permute_282: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_66: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_282, [16, 32, 128, 80]);  permute_282 = None
        clone_123: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_511: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [512, 128, 80]);  clone_123 = None
        bmm_31: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_510, view_511);  view_510 = view_511 = None
        view_512: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [16, 32, 128, 80]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_284: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_124: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_513: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [16, 128, -1]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_514: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_513, [2048, 2560]);  view_513 = None
        permute_285: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
        addmm_149: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg396_1, view_514, permute_285);  arg396_1 = view_514 = permute_285 = None
        view_515: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_149, [16, 128, 2560]);  addmm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_176: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, view_515);  add_172 = view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_668: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_176, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_668, [2], correction = 0, keepdim = True)
        getitem_218: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_46[0]
        getitem_219: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        sub_62: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_668, getitem_219);  convert_element_type_668 = getitem_219 = None
        add_177: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
        rsqrt_46: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        mul_171: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_46);  sub_62 = rsqrt_46 = None
        mul_172: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, arg397_1);  mul_171 = arg397_1 = None
        add_178: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, arg398_1);  mul_172 = arg398_1 = None
        convert_element_type_669: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_178, torch.bfloat16);  add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_516: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_669, [2048, 2560]);  convert_element_type_669 = None
        permute_286: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
        addmm_150: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg400_1, view_516, permute_286);  arg400_1 = view_516 = permute_286 = None
        view_517: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_150, [16, 128, 10240]);  addmm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_673: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_517, torch.float32);  view_517 = None
        mul_173: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_673, 0.5)
        mul_174: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_673, 0.7071067811865476);  convert_element_type_673 = None
        erf_15: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_179: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_175: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, add_179);  mul_173 = add_179 = None
        convert_element_type_674: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_518: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_674, [2048, 10240]);  convert_element_type_674 = None
        permute_287: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg401_1, [1, 0]);  arg401_1 = None
        addmm_151: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg402_1, view_518, permute_287);  arg402_1 = view_518 = permute_287 = None
        view_519: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_151, [16, 128, 2560]);  addmm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_180: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_176, view_519);  add_176 = view_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_678: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_180, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_678, [2], correction = 0, keepdim = True)
        getitem_220: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_47[0]
        getitem_221: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        sub_63: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_678, getitem_221);  convert_element_type_678 = getitem_221 = None
        add_181: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_47: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        mul_176: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_47);  sub_63 = rsqrt_47 = None
        mul_177: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, arg403_1);  mul_176 = arg403_1 = None
        add_182: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_177, arg404_1);  mul_177 = arg404_1 = None
        convert_element_type_679: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_182, torch.bfloat16);  add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_520: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_679, [2048, 2560])
        permute_288: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        addmm_152: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg406_1, view_520, permute_288);  arg406_1 = view_520 = permute_288 = None
        view_521: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_152, [16, 128, 2560]);  addmm_152 = None
        view_522: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_521, [16, 128, -1, 80]);  view_521 = None
        permute_289: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_523: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_679, [2048, 2560])
        permute_290: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
        addmm_153: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg408_1, view_523, permute_290);  arg408_1 = view_523 = permute_290 = None
        view_524: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_153, [16, 128, 2560]);  addmm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_527: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_524, [16, 128, -1, 80]);  view_524 = None
        permute_292: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_525: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_679, [2048, 2560]);  convert_element_type_679 = None
        permute_291: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        addmm_154: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg410_1, view_525, permute_291);  arg410_1 = view_525 = permute_291 = None
        view_526: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_154, [16, 128, 2560]);  addmm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_528: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_526, [16, 128, -1, 80]);  view_526 = None
        permute_293: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_61: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_46: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_61, full_default_60);  full_default_61 = full_default_60 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_289, permute_292, permute_293, where_46, False, scale = 0.11180339887498948);  permute_289 = permute_292 = permute_293 = where_46 = None
        getitem_222: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_294: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_222, [0, 2, 1, 3]);  getitem_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_529: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_294, [16, 128, -1]);  permute_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_530: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_529, [2048, 2560]);  view_529 = None
        permute_295: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
        addmm_155: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg412_1, view_530, permute_295);  arg412_1 = view_530 = permute_295 = None
        view_531: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_155, [16, 128, 2560]);  addmm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_183: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_180, view_531);  add_180 = view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_692: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_183, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_692, [2], correction = 0, keepdim = True)
        getitem_231: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_48[0]
        getitem_232: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_62: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_62 = None
        scalar_tensor_63: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_63 = None
        full_default_62: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_64: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_692, getitem_232);  convert_element_type_692 = getitem_232 = None
        add_184: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_231, 1e-05);  getitem_231 = None
        rsqrt_48: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        mul_178: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_48);  sub_64 = rsqrt_48 = None
        mul_179: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, arg413_1);  mul_178 = arg413_1 = None
        add_185: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, arg414_1);  mul_179 = arg414_1 = None
        convert_element_type_693: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_185, torch.bfloat16);  add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_532: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_693, [2048, 2560]);  convert_element_type_693 = None
        permute_296: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
        addmm_156: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg416_1, view_532, permute_296);  arg416_1 = view_532 = permute_296 = None
        view_533: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_156, [16, 128, 2560]);  addmm_156 = None
        view_534: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_533, [16, 128, -1, 80]);  view_533 = None
        permute_297: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_180: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_297, 0.334370152488211);  permute_297 = None
        expand_67: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_180, [16, 32, 128, 80]);  mul_180 = None
        clone_129: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_541: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [512, 128, 80]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_535: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_298: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg417_1, [1, 0]);  arg417_1 = None
        addmm_157: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg418_1, view_535, permute_298);  arg418_1 = view_535 = permute_298 = None
        view_536: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_157, [16, 128, 2560]);  addmm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_539: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_536, [16, 128, -1, 80]);  view_536 = None
        permute_300: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_302: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_300, [0, 1, 3, 2]);  permute_300 = None
        mul_181: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_302, 0.334370152488211);  permute_302 = None
        expand_68: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_181, [16, 32, 80, 128]);  mul_181 = None
        clone_130: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_542: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [512, 80, 128]);  clone_130 = None
        bmm_32: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_541, view_542);  view_541 = view_542 = None
        view_543: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [16, 32, 128, 128]);  bmm_32 = None
        eq_16: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_543, -inf)
        logical_not_32: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_16);  eq_16 = None
        any_17: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_32, -1, True);  logical_not_32 = None
        logical_not_33: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        full_default_63: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_705: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_543, torch.float32);  view_543 = None
        amax_16: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_705, [-1], True)
        sub_65: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_705, amax_16);  convert_element_type_705 = amax_16 = None
        exp_16: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
        sum_17: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_706: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        where_48: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_33, full_default_63, convert_element_type_706);  logical_not_33 = full_default_63 = convert_element_type_706 = None
        expand_69: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_48, [16, 32, 128, 128]);  where_48 = None
        view_544: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_69, [512, 128, 128]);  expand_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_537: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_299: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
        addmm_158: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg420_1, view_537, permute_299);  arg420_1 = view_537 = permute_299 = None
        view_538: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_158, [16, 128, 2560]);  addmm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_540: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_538, [16, 128, -1, 80]);  view_538 = None
        permute_301: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_70: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_301, [16, 32, 128, 80]);  permute_301 = None
        clone_131: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_545: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [512, 128, 80]);  clone_131 = None
        bmm_33: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_544, view_545);  view_544 = view_545 = None
        view_546: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [16, 32, 128, 80]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_303: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
        clone_132: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_547: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_132, [16, 128, -1]);  clone_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_548: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_547, [2048, 2560]);  view_547 = None
        permute_304: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
        addmm_159: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg422_1, view_548, permute_304);  arg422_1 = view_548 = permute_304 = None
        view_549: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_159, [16, 128, 2560]);  addmm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_187: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_183, view_549);  add_183 = view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_712: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_187, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_712, [2], correction = 0, keepdim = True)
        getitem_233: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_49[0]
        getitem_234: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        sub_66: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_712, getitem_234);  convert_element_type_712 = getitem_234 = None
        add_188: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_233, 1e-05);  getitem_233 = None
        rsqrt_49: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_182: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_49);  sub_66 = rsqrt_49 = None
        mul_183: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, arg423_1);  mul_182 = arg423_1 = None
        add_189: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_183, arg424_1);  mul_183 = arg424_1 = None
        convert_element_type_713: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_550: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_713, [2048, 2560]);  convert_element_type_713 = None
        permute_305: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
        addmm_160: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg426_1, view_550, permute_305);  arg426_1 = view_550 = permute_305 = None
        view_551: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_160, [16, 128, 10240]);  addmm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_717: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_551, torch.float32);  view_551 = None
        mul_184: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_717, 0.5)
        mul_185: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_717, 0.7071067811865476);  convert_element_type_717 = None
        erf_16: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_190: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_186: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, add_190);  mul_184 = add_190 = None
        convert_element_type_718: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_552: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_718, [2048, 10240]);  convert_element_type_718 = None
        permute_306: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg427_1, [1, 0]);  arg427_1 = None
        addmm_161: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg428_1, view_552, permute_306);  arg428_1 = view_552 = permute_306 = None
        view_553: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_161, [16, 128, 2560]);  addmm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_191: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_187, view_553);  add_187 = view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_722: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_722, [2], correction = 0, keepdim = True)
        getitem_235: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_50[0]
        getitem_236: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        sub_67: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_722, getitem_236);  convert_element_type_722 = getitem_236 = None
        add_192: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_235, 1e-05);  getitem_235 = None
        rsqrt_50: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_187: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_50);  sub_67 = rsqrt_50 = None
        mul_188: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, arg429_1);  mul_187 = arg429_1 = None
        add_193: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, arg430_1);  mul_188 = arg430_1 = None
        convert_element_type_723: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16);  add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_554: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_723, [2048, 2560])
        permute_307: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        addmm_162: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg432_1, view_554, permute_307);  arg432_1 = view_554 = permute_307 = None
        view_555: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_162, [16, 128, 2560]);  addmm_162 = None
        view_556: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_555, [16, 128, -1, 80]);  view_555 = None
        permute_308: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_557: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_723, [2048, 2560])
        permute_309: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg433_1, [1, 0]);  arg433_1 = None
        addmm_163: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg434_1, view_557, permute_309);  arg434_1 = view_557 = permute_309 = None
        view_558: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_163, [16, 128, 2560]);  addmm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_561: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_558, [16, 128, -1, 80]);  view_558 = None
        permute_311: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_559: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_723, [2048, 2560]);  convert_element_type_723 = None
        permute_310: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_164: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg436_1, view_559, permute_310);  arg436_1 = view_559 = permute_310 = None
        view_560: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_164, [16, 128, 2560]);  addmm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_562: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_560, [16, 128, -1, 80]);  view_560 = None
        permute_312: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_65: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_49: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_65, full_default_64);  full_default_65 = full_default_64 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_308, permute_311, permute_312, where_49, False, scale = 0.11180339887498948);  permute_308 = permute_311 = permute_312 = where_49 = None
        getitem_237: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_313: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_237, [0, 2, 1, 3]);  getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_563: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_313, [16, 128, -1]);  permute_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_564: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_563, [2048, 2560]);  view_563 = None
        permute_314: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg437_1, [1, 0]);  arg437_1 = None
        addmm_165: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg438_1, view_564, permute_314);  arg438_1 = view_564 = permute_314 = None
        view_565: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_165, [16, 128, 2560]);  addmm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_194: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_191, view_565);  add_191 = view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_736: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_194, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_736, [2], correction = 0, keepdim = True)
        getitem_246: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_51[0]
        getitem_247: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_66: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_66 = None
        scalar_tensor_67: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_67 = None
        full_default_66: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_68: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_736, getitem_247);  convert_element_type_736 = getitem_247 = None
        add_195: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_51: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        mul_189: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_51);  sub_68 = rsqrt_51 = None
        mul_190: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, arg439_1);  mul_189 = arg439_1 = None
        add_196: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_190, arg440_1);  mul_190 = arg440_1 = None
        convert_element_type_737: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_566: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_737, [2048, 2560]);  convert_element_type_737 = None
        permute_315: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_166: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg442_1, view_566, permute_315);  arg442_1 = view_566 = permute_315 = None
        view_567: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_166, [16, 128, 2560]);  addmm_166 = None
        view_568: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_567, [16, 128, -1, 80]);  view_567 = None
        permute_316: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_191: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_316, 0.334370152488211);  permute_316 = None
        expand_71: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_191, [16, 32, 128, 80]);  mul_191 = None
        clone_137: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_575: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [512, 128, 80]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_569: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_317: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
        addmm_167: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg444_1, view_569, permute_317);  arg444_1 = view_569 = permute_317 = None
        view_570: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_167, [16, 128, 2560]);  addmm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_573: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_570, [16, 128, -1, 80]);  view_570 = None
        permute_319: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_321: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_319, [0, 1, 3, 2]);  permute_319 = None
        mul_192: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_321, 0.334370152488211);  permute_321 = None
        expand_72: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_192, [16, 32, 80, 128]);  mul_192 = None
        clone_138: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_576: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [512, 80, 128]);  clone_138 = None
        bmm_34: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_575, view_576);  view_575 = view_576 = None
        view_577: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [16, 32, 128, 128]);  bmm_34 = None
        eq_17: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_577, -inf)
        logical_not_34: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_17);  eq_17 = None
        any_18: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_34, -1, True);  logical_not_34 = None
        logical_not_35: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        full_default_67: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_749: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_577, torch.float32);  view_577 = None
        amax_17: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_749, [-1], True)
        sub_69: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_749, amax_17);  convert_element_type_749 = amax_17 = None
        exp_17: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
        sum_18: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_750: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        where_51: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_35, full_default_67, convert_element_type_750);  logical_not_35 = full_default_67 = convert_element_type_750 = None
        expand_73: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_51, [16, 32, 128, 128]);  where_51 = None
        view_578: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_73, [512, 128, 128]);  expand_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_571: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_318: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
        addmm_168: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg446_1, view_571, permute_318);  arg446_1 = view_571 = permute_318 = None
        view_572: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_168, [16, 128, 2560]);  addmm_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_574: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_572, [16, 128, -1, 80]);  view_572 = None
        permute_320: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_74: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_320, [16, 32, 128, 80]);  permute_320 = None
        clone_139: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_579: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_139, [512, 128, 80]);  clone_139 = None
        bmm_35: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_578, view_579);  view_578 = view_579 = None
        view_580: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [16, 32, 128, 80]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_322: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
        clone_140: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_581: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_140, [16, 128, -1]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_582: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_581, [2048, 2560]);  view_581 = None
        permute_323: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        addmm_169: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg448_1, view_582, permute_323);  arg448_1 = view_582 = permute_323 = None
        view_583: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_169, [16, 128, 2560]);  addmm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_198: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_194, view_583);  add_194 = view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_756: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_198, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_756, [2], correction = 0, keepdim = True)
        getitem_248: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_52[0]
        getitem_249: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        sub_70: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_756, getitem_249);  convert_element_type_756 = getitem_249 = None
        add_199: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_248, 1e-05);  getitem_248 = None
        rsqrt_52: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        mul_193: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_52);  sub_70 = rsqrt_52 = None
        mul_194: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_193, arg449_1);  mul_193 = arg449_1 = None
        add_200: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_194, arg450_1);  mul_194 = arg450_1 = None
        convert_element_type_757: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_200, torch.bfloat16);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_584: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_757, [2048, 2560]);  convert_element_type_757 = None
        permute_324: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg451_1, [1, 0]);  arg451_1 = None
        addmm_170: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg452_1, view_584, permute_324);  arg452_1 = view_584 = permute_324 = None
        view_585: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_170, [16, 128, 10240]);  addmm_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_761: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_585, torch.float32);  view_585 = None
        mul_195: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_761, 0.5)
        mul_196: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_761, 0.7071067811865476);  convert_element_type_761 = None
        erf_17: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_196);  mul_196 = None
        add_201: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_197: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_195, add_201);  mul_195 = add_201 = None
        convert_element_type_762: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_197, torch.bfloat16);  mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_586: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_762, [2048, 10240]);  convert_element_type_762 = None
        permute_325: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg453_1, [1, 0]);  arg453_1 = None
        addmm_171: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg454_1, view_586, permute_325);  arg454_1 = view_586 = permute_325 = None
        view_587: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_171, [16, 128, 2560]);  addmm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_202: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_198, view_587);  add_198 = view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_766: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_766, [2], correction = 0, keepdim = True)
        getitem_250: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_53[0]
        getitem_251: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_53[1];  var_mean_53 = None
        sub_71: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_766, getitem_251);  convert_element_type_766 = getitem_251 = None
        add_203: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_250, 1e-05);  getitem_250 = None
        rsqrt_53: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_198: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_53);  sub_71 = rsqrt_53 = None
        mul_199: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, arg455_1);  mul_198 = arg455_1 = None
        add_204: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, arg456_1);  mul_199 = arg456_1 = None
        convert_element_type_767: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_588: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_767, [2048, 2560])
        permute_326: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
        addmm_172: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg458_1, view_588, permute_326);  arg458_1 = view_588 = permute_326 = None
        view_589: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_172, [16, 128, 2560]);  addmm_172 = None
        view_590: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_589, [16, 128, -1, 80]);  view_589 = None
        permute_327: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_590, [0, 2, 1, 3]);  view_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_591: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_767, [2048, 2560])
        permute_328: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
        addmm_173: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg460_1, view_591, permute_328);  arg460_1 = view_591 = permute_328 = None
        view_592: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_173, [16, 128, 2560]);  addmm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_595: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_592, [16, 128, -1, 80]);  view_592 = None
        permute_330: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_595, [0, 2, 1, 3]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_593: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_767, [2048, 2560]);  convert_element_type_767 = None
        permute_329: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        addmm_174: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg462_1, view_593, permute_329);  arg462_1 = view_593 = permute_329 = None
        view_594: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_174, [16, 128, 2560]);  addmm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_596: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_594, [16, 128, -1, 80]);  view_594 = None
        permute_331: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_69: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_52: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_69, full_default_68);  full_default_69 = full_default_68 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_327, permute_330, permute_331, where_52, False, scale = 0.11180339887498948);  permute_327 = permute_330 = permute_331 = where_52 = None
        getitem_252: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_16[0];  _scaled_dot_product_cudnn_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_332: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3]);  getitem_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_597: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_332, [16, 128, -1]);  permute_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_598: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_597, [2048, 2560]);  view_597 = None
        permute_333: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
        addmm_175: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg464_1, view_598, permute_333);  arg464_1 = view_598 = permute_333 = None
        view_599: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_175, [16, 128, 2560]);  addmm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_205: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_202, view_599);  add_202 = view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_780: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_205, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_780, [2], correction = 0, keepdim = True)
        getitem_261: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_54[0]
        getitem_262: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_54[1];  var_mean_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_70: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_70 = None
        scalar_tensor_71: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_71 = None
        full_default_70: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_72: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_780, getitem_262);  convert_element_type_780 = getitem_262 = None
        add_206: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_261, 1e-05);  getitem_261 = None
        rsqrt_54: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        mul_200: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_54);  sub_72 = rsqrt_54 = None
        mul_201: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, arg465_1);  mul_200 = arg465_1 = None
        add_207: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_201, arg466_1);  mul_201 = arg466_1 = None
        convert_element_type_781: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_207, torch.bfloat16);  add_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_600: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_781, [2048, 2560]);  convert_element_type_781 = None
        permute_334: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg467_1, [1, 0]);  arg467_1 = None
        addmm_176: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg468_1, view_600, permute_334);  arg468_1 = view_600 = permute_334 = None
        view_601: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_176, [16, 128, 2560]);  addmm_176 = None
        view_602: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_601, [16, 128, -1, 80]);  view_601 = None
        permute_335: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_602, [0, 2, 1, 3]);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_202: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_335, 0.334370152488211);  permute_335 = None
        expand_75: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_202, [16, 32, 128, 80]);  mul_202 = None
        clone_145: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_609: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [512, 128, 80]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_603: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_336: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg469_1, [1, 0]);  arg469_1 = None
        addmm_177: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg470_1, view_603, permute_336);  arg470_1 = view_603 = permute_336 = None
        view_604: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_177, [16, 128, 2560]);  addmm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_607: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_604, [16, 128, -1, 80]);  view_604 = None
        permute_338: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_340: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_338, [0, 1, 3, 2]);  permute_338 = None
        mul_203: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_340, 0.334370152488211);  permute_340 = None
        expand_76: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_203, [16, 32, 80, 128]);  mul_203 = None
        clone_146: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_610: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [512, 80, 128]);  clone_146 = None
        bmm_36: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_609, view_610);  view_609 = view_610 = None
        view_611: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [16, 32, 128, 128]);  bmm_36 = None
        eq_18: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_611, -inf)
        logical_not_36: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_18);  eq_18 = None
        any_19: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_36, -1, True);  logical_not_36 = None
        logical_not_37: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        full_default_71: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_793: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_611, torch.float32);  view_611 = None
        amax_18: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_793, [-1], True)
        sub_73: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_793, amax_18);  convert_element_type_793 = amax_18 = None
        exp_18: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
        sum_19: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_794: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        where_54: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_37, full_default_71, convert_element_type_794);  logical_not_37 = full_default_71 = convert_element_type_794 = None
        expand_77: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_54, [16, 32, 128, 128]);  where_54 = None
        view_612: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_77, [512, 128, 128]);  expand_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_605: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_337: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
        addmm_178: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg472_1, view_605, permute_337);  arg472_1 = view_605 = permute_337 = None
        view_606: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_178, [16, 128, 2560]);  addmm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_608: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_606, [16, 128, -1, 80]);  view_606 = None
        permute_339: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_78: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_339, [16, 32, 128, 80]);  permute_339 = None
        clone_147: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_613: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_147, [512, 128, 80]);  clone_147 = None
        bmm_37: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_612, view_613);  view_612 = view_613 = None
        view_614: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [16, 32, 128, 80]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_341: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        clone_148: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_341, memory_format = torch.contiguous_format);  permute_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_615: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_148, [16, 128, -1]);  clone_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_616: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_615, [2048, 2560]);  view_615 = None
        permute_342: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg473_1, [1, 0]);  arg473_1 = None
        addmm_179: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg474_1, view_616, permute_342);  arg474_1 = view_616 = permute_342 = None
        view_617: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_179, [16, 128, 2560]);  addmm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_209: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_205, view_617);  add_205 = view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_800: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_209, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_800, [2], correction = 0, keepdim = True)
        getitem_263: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_55[0]
        getitem_264: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_55[1];  var_mean_55 = None
        sub_74: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_800, getitem_264);  convert_element_type_800 = getitem_264 = None
        add_210: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_263, 1e-05);  getitem_263 = None
        rsqrt_55: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        mul_204: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_55);  sub_74 = rsqrt_55 = None
        mul_205: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, arg475_1);  mul_204 = arg475_1 = None
        add_211: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_205, arg476_1);  mul_205 = arg476_1 = None
        convert_element_type_801: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_211, torch.bfloat16);  add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_618: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_801, [2048, 2560]);  convert_element_type_801 = None
        permute_343: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_180: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg478_1, view_618, permute_343);  arg478_1 = view_618 = permute_343 = None
        view_619: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_180, [16, 128, 10240]);  addmm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_805: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_619, torch.float32);  view_619 = None
        mul_206: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_805, 0.5)
        mul_207: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_805, 0.7071067811865476);  convert_element_type_805 = None
        erf_18: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_212: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_208: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, add_212);  mul_206 = add_212 = None
        convert_element_type_806: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_620: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_806, [2048, 10240]);  convert_element_type_806 = None
        permute_344: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
        addmm_181: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg480_1, view_620, permute_344);  arg480_1 = view_620 = permute_344 = None
        view_621: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_181, [16, 128, 2560]);  addmm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_213: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_209, view_621);  add_209 = view_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_810: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_213, torch.float32)
        var_mean_56 = torch.ops.aten.var_mean.correction(convert_element_type_810, [2], correction = 0, keepdim = True)
        getitem_265: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_56[0]
        getitem_266: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_56[1];  var_mean_56 = None
        sub_75: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_810, getitem_266);  convert_element_type_810 = getitem_266 = None
        add_214: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_265, 1e-05);  getitem_265 = None
        rsqrt_56: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        mul_209: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_56);  sub_75 = rsqrt_56 = None
        mul_210: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_209, arg481_1);  mul_209 = arg481_1 = None
        add_215: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_210, arg482_1);  mul_210 = arg482_1 = None
        convert_element_type_811: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_622: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_811, [2048, 2560])
        permute_345: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
        addmm_182: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg484_1, view_622, permute_345);  arg484_1 = view_622 = permute_345 = None
        view_623: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_182, [16, 128, 2560]);  addmm_182 = None
        view_624: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_623, [16, 128, -1, 80]);  view_623 = None
        permute_346: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_624, [0, 2, 1, 3]);  view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_625: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_811, [2048, 2560])
        permute_347: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg485_1, [1, 0]);  arg485_1 = None
        addmm_183: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg486_1, view_625, permute_347);  arg486_1 = view_625 = permute_347 = None
        view_626: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_183, [16, 128, 2560]);  addmm_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_629: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_626, [16, 128, -1, 80]);  view_626 = None
        permute_349: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_627: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_811, [2048, 2560]);  convert_element_type_811 = None
        permute_348: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
        addmm_184: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg488_1, view_627, permute_348);  arg488_1 = view_627 = permute_348 = None
        view_628: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_184, [16, 128, 2560]);  addmm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_630: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_628, [16, 128, -1, 80]);  view_628 = None
        permute_350: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_73: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_72: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_55: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_73, full_default_72);  full_default_73 = full_default_72 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_346, permute_349, permute_350, where_55, False, scale = 0.11180339887498948);  permute_346 = permute_349 = permute_350 = where_55 = None
        getitem_267: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_17[0];  _scaled_dot_product_cudnn_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_351: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_267, [0, 2, 1, 3]);  getitem_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_631: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_351, [16, 128, -1]);  permute_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_632: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_631, [2048, 2560]);  view_631 = None
        permute_352: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
        addmm_185: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg490_1, view_632, permute_352);  arg490_1 = view_632 = permute_352 = None
        view_633: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_185, [16, 128, 2560]);  addmm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_216: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_213, view_633);  add_213 = view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_824: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_216, torch.float32)
        var_mean_57 = torch.ops.aten.var_mean.correction(convert_element_type_824, [2], correction = 0, keepdim = True)
        getitem_276: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_57[0]
        getitem_277: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_57[1];  var_mean_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_74: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_74 = None
        scalar_tensor_75: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_75 = None
        full_default_74: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_76: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_824, getitem_277);  convert_element_type_824 = getitem_277 = None
        add_217: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_276, 1e-05);  getitem_276 = None
        rsqrt_57: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        mul_211: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_57);  sub_76 = rsqrt_57 = None
        mul_212: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, arg491_1);  mul_211 = arg491_1 = None
        add_218: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_212, arg492_1);  mul_212 = arg492_1 = None
        convert_element_type_825: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_218, torch.bfloat16);  add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_634: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_825, [2048, 2560]);  convert_element_type_825 = None
        permute_353: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_186: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg494_1, view_634, permute_353);  arg494_1 = view_634 = permute_353 = None
        view_635: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_186, [16, 128, 2560]);  addmm_186 = None
        view_636: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_635, [16, 128, -1, 80]);  view_635 = None
        permute_354: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_213: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_354, 0.334370152488211);  permute_354 = None
        expand_79: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_213, [16, 32, 128, 80]);  mul_213 = None
        clone_153: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_643: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [512, 128, 80]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_637: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_355: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
        addmm_187: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg496_1, view_637, permute_355);  arg496_1 = view_637 = permute_355 = None
        view_638: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_187, [16, 128, 2560]);  addmm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_641: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_638, [16, 128, -1, 80]);  view_638 = None
        permute_357: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_641, [0, 2, 1, 3]);  view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_359: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_357, [0, 1, 3, 2]);  permute_357 = None
        mul_214: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_359, 0.334370152488211);  permute_359 = None
        expand_80: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_214, [16, 32, 80, 128]);  mul_214 = None
        clone_154: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_644: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [512, 80, 128]);  clone_154 = None
        bmm_38: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_643, view_644);  view_643 = view_644 = None
        view_645: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [16, 32, 128, 128]);  bmm_38 = None
        eq_19: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_645, -inf)
        logical_not_38: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_19);  eq_19 = None
        any_20: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_38, -1, True);  logical_not_38 = None
        logical_not_39: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        full_default_75: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_837: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_645, torch.float32);  view_645 = None
        amax_19: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_837, [-1], True)
        sub_77: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_837, amax_19);  convert_element_type_837 = amax_19 = None
        exp_19: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_77);  sub_77 = None
        sum_20: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_838: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        where_57: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_39, full_default_75, convert_element_type_838);  logical_not_39 = full_default_75 = convert_element_type_838 = None
        expand_81: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_57, [16, 32, 128, 128]);  where_57 = None
        view_646: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_81, [512, 128, 128]);  expand_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_639: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_356: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
        addmm_188: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg498_1, view_639, permute_356);  arg498_1 = view_639 = permute_356 = None
        view_640: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_188, [16, 128, 2560]);  addmm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_642: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_640, [16, 128, -1, 80]);  view_640 = None
        permute_358: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_642, [0, 2, 1, 3]);  view_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_82: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_358, [16, 32, 128, 80]);  permute_358 = None
        clone_155: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_647: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_155, [512, 128, 80]);  clone_155 = None
        bmm_39: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_646, view_647);  view_646 = view_647 = None
        view_648: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [16, 32, 128, 80]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_360: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None
        clone_156: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_649: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_156, [16, 128, -1]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_650: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_649, [2048, 2560]);  view_649 = None
        permute_361: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        addmm_189: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg500_1, view_650, permute_361);  arg500_1 = view_650 = permute_361 = None
        view_651: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_189, [16, 128, 2560]);  addmm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_220: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_216, view_651);  add_216 = view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_844: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_220, torch.float32)
        var_mean_58 = torch.ops.aten.var_mean.correction(convert_element_type_844, [2], correction = 0, keepdim = True)
        getitem_278: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_58[0]
        getitem_279: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_58[1];  var_mean_58 = None
        sub_78: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_844, getitem_279);  convert_element_type_844 = getitem_279 = None
        add_221: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_278, 1e-05);  getitem_278 = None
        rsqrt_58: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        mul_215: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_58);  sub_78 = rsqrt_58 = None
        mul_216: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_215, arg501_1);  mul_215 = arg501_1 = None
        add_222: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_216, arg502_1);  mul_216 = arg502_1 = None
        convert_element_type_845: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_222, torch.bfloat16);  add_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_652: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_845, [2048, 2560]);  convert_element_type_845 = None
        permute_362: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg503_1, [1, 0]);  arg503_1 = None
        addmm_190: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg504_1, view_652, permute_362);  arg504_1 = view_652 = permute_362 = None
        view_653: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_190, [16, 128, 10240]);  addmm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_849: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_653, torch.float32);  view_653 = None
        mul_217: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_849, 0.5)
        mul_218: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_849, 0.7071067811865476);  convert_element_type_849 = None
        erf_19: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
        add_223: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_219: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, add_223);  mul_217 = add_223 = None
        convert_element_type_850: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_219, torch.bfloat16);  mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_654: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_850, [2048, 10240]);  convert_element_type_850 = None
        permute_363: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg505_1, [1, 0]);  arg505_1 = None
        addmm_191: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg506_1, view_654, permute_363);  arg506_1 = view_654 = permute_363 = None
        view_655: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_191, [16, 128, 2560]);  addmm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_224: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_220, view_655);  add_220 = view_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_854: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_224, torch.float32)
        var_mean_59 = torch.ops.aten.var_mean.correction(convert_element_type_854, [2], correction = 0, keepdim = True)
        getitem_280: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_59[0]
        getitem_281: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_59[1];  var_mean_59 = None
        sub_79: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_854, getitem_281);  convert_element_type_854 = getitem_281 = None
        add_225: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_280, 1e-05);  getitem_280 = None
        rsqrt_59: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_220: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_59);  sub_79 = rsqrt_59 = None
        mul_221: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, arg507_1);  mul_220 = arg507_1 = None
        add_226: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, arg508_1);  mul_221 = arg508_1 = None
        convert_element_type_855: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_656: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_855, [2048, 2560])
        permute_364: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        addmm_192: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg510_1, view_656, permute_364);  arg510_1 = view_656 = permute_364 = None
        view_657: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_192, [16, 128, 2560]);  addmm_192 = None
        view_658: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_657, [16, 128, -1, 80]);  view_657 = None
        permute_365: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_659: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_855, [2048, 2560])
        permute_366: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg511_1, [1, 0]);  arg511_1 = None
        addmm_193: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg512_1, view_659, permute_366);  arg512_1 = view_659 = permute_366 = None
        view_660: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_193, [16, 128, 2560]);  addmm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_663: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_660, [16, 128, -1, 80]);  view_660 = None
        permute_368: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_663, [0, 2, 1, 3]);  view_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_661: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_855, [2048, 2560]);  convert_element_type_855 = None
        permute_367: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg513_1, [1, 0]);  arg513_1 = None
        addmm_194: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg514_1, view_661, permute_367);  arg514_1 = view_661 = permute_367 = None
        view_662: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_194, [16, 128, 2560]);  addmm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_664: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_662, [16, 128, -1, 80]);  view_662 = None
        permute_369: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_77: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_58: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_77, full_default_76);  full_default_77 = full_default_76 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_365, permute_368, permute_369, where_58, False, scale = 0.11180339887498948);  permute_365 = permute_368 = permute_369 = where_58 = None
        getitem_282: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_18[0];  _scaled_dot_product_cudnn_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_370: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_282, [0, 2, 1, 3]);  getitem_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_665: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_370, [16, 128, -1]);  permute_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_666: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_665, [2048, 2560]);  view_665 = None
        permute_371: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
        addmm_195: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg516_1, view_666, permute_371);  arg516_1 = view_666 = permute_371 = None
        view_667: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_195, [16, 128, 2560]);  addmm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_227: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_224, view_667);  add_224 = view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_868: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_227, torch.float32)
        var_mean_60 = torch.ops.aten.var_mean.correction(convert_element_type_868, [2], correction = 0, keepdim = True)
        getitem_291: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_60[0]
        getitem_292: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_60[1];  var_mean_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_78: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_78 = None
        scalar_tensor_79: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_79 = None
        full_default_78: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_80: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_868, getitem_292);  convert_element_type_868 = getitem_292 = None
        add_228: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_291, 1e-05);  getitem_291 = None
        rsqrt_60: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        mul_222: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_60);  sub_80 = rsqrt_60 = None
        mul_223: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, arg517_1);  mul_222 = arg517_1 = None
        add_229: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_223, arg518_1);  mul_223 = arg518_1 = None
        convert_element_type_869: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_229, torch.bfloat16);  add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_668: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_869, [2048, 2560]);  convert_element_type_869 = None
        permute_372: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg519_1, [1, 0]);  arg519_1 = None
        addmm_196: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg520_1, view_668, permute_372);  arg520_1 = view_668 = permute_372 = None
        view_669: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_196, [16, 128, 2560]);  addmm_196 = None
        view_670: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_669, [16, 128, -1, 80]);  view_669 = None
        permute_373: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_670, [0, 2, 1, 3]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_224: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_373, 0.334370152488211);  permute_373 = None
        expand_83: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_224, [16, 32, 128, 80]);  mul_224 = None
        clone_161: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_677: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [512, 128, 80]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_671: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_374: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg521_1, [1, 0]);  arg521_1 = None
        addmm_197: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg522_1, view_671, permute_374);  arg522_1 = view_671 = permute_374 = None
        view_672: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_197, [16, 128, 2560]);  addmm_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_675: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_672, [16, 128, -1, 80]);  view_672 = None
        permute_376: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_675, [0, 2, 1, 3]);  view_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_378: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_376, [0, 1, 3, 2]);  permute_376 = None
        mul_225: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_378, 0.334370152488211);  permute_378 = None
        expand_84: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_225, [16, 32, 80, 128]);  mul_225 = None
        clone_162: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_678: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [512, 80, 128]);  clone_162 = None
        bmm_40: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_677, view_678);  view_677 = view_678 = None
        view_679: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [16, 32, 128, 128]);  bmm_40 = None
        eq_20: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_679, -inf)
        logical_not_40: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_20);  eq_20 = None
        any_21: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_40, -1, True);  logical_not_40 = None
        logical_not_41: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        full_default_79: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_881: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_679, torch.float32);  view_679 = None
        amax_20: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_881, [-1], True)
        sub_81: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_881, amax_20);  convert_element_type_881 = amax_20 = None
        exp_20: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_81);  sub_81 = None
        sum_21: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_882: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        where_60: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_41, full_default_79, convert_element_type_882);  logical_not_41 = full_default_79 = convert_element_type_882 = None
        expand_85: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_60, [16, 32, 128, 128]);  where_60 = None
        view_680: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_85, [512, 128, 128]);  expand_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_673: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_375: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg523_1, [1, 0]);  arg523_1 = None
        addmm_198: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg524_1, view_673, permute_375);  arg524_1 = view_673 = permute_375 = None
        view_674: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_198, [16, 128, 2560]);  addmm_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_676: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_674, [16, 128, -1, 80]);  view_674 = None
        permute_377: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_86: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_377, [16, 32, 128, 80]);  permute_377 = None
        clone_163: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_681: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_163, [512, 128, 80]);  clone_163 = None
        bmm_41: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_680, view_681);  view_680 = view_681 = None
        view_682: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [16, 32, 128, 80]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_379: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
        clone_164: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_379, memory_format = torch.contiguous_format);  permute_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_683: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_164, [16, 128, -1]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_684: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_683, [2048, 2560]);  view_683 = None
        permute_380: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg525_1, [1, 0]);  arg525_1 = None
        addmm_199: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg526_1, view_684, permute_380);  arg526_1 = view_684 = permute_380 = None
        view_685: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_199, [16, 128, 2560]);  addmm_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_231: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_227, view_685);  add_227 = view_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_888: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_231, torch.float32)
        var_mean_61 = torch.ops.aten.var_mean.correction(convert_element_type_888, [2], correction = 0, keepdim = True)
        getitem_293: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_61[0]
        getitem_294: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_61[1];  var_mean_61 = None
        sub_82: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_888, getitem_294);  convert_element_type_888 = getitem_294 = None
        add_232: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_293, 1e-05);  getitem_293 = None
        rsqrt_61: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        mul_226: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_61);  sub_82 = rsqrt_61 = None
        mul_227: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_226, arg527_1);  mul_226 = arg527_1 = None
        add_233: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_227, arg528_1);  mul_227 = arg528_1 = None
        convert_element_type_889: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_233, torch.bfloat16);  add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_686: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_889, [2048, 2560]);  convert_element_type_889 = None
        permute_381: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg529_1, [1, 0]);  arg529_1 = None
        addmm_200: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg530_1, view_686, permute_381);  arg530_1 = view_686 = permute_381 = None
        view_687: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_200, [16, 128, 10240]);  addmm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_893: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_687, torch.float32);  view_687 = None
        mul_228: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_893, 0.5)
        mul_229: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_893, 0.7071067811865476);  convert_element_type_893 = None
        erf_20: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_229);  mul_229 = None
        add_234: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_230: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_228, add_234);  mul_228 = add_234 = None
        convert_element_type_894: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_230, torch.bfloat16);  mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_688: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_894, [2048, 10240]);  convert_element_type_894 = None
        permute_382: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg531_1, [1, 0]);  arg531_1 = None
        addmm_201: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg532_1, view_688, permute_382);  arg532_1 = view_688 = permute_382 = None
        view_689: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_201, [16, 128, 2560]);  addmm_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_235: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_231, view_689);  add_231 = view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_898: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_235, torch.float32)
        var_mean_62 = torch.ops.aten.var_mean.correction(convert_element_type_898, [2], correction = 0, keepdim = True)
        getitem_295: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_62[0]
        getitem_296: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_62[1];  var_mean_62 = None
        sub_83: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_898, getitem_296);  convert_element_type_898 = getitem_296 = None
        add_236: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_295, 1e-05);  getitem_295 = None
        rsqrt_62: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_231: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_62);  sub_83 = rsqrt_62 = None
        mul_232: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, arg533_1);  mul_231 = arg533_1 = None
        add_237: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_232, arg534_1);  mul_232 = arg534_1 = None
        convert_element_type_899: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_690: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_899, [2048, 2560])
        permute_383: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg535_1, [1, 0]);  arg535_1 = None
        addmm_202: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg536_1, view_690, permute_383);  arg536_1 = view_690 = permute_383 = None
        view_691: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_202, [16, 128, 2560]);  addmm_202 = None
        view_692: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_691, [16, 128, -1, 80]);  view_691 = None
        permute_384: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_693: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_899, [2048, 2560])
        permute_385: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg537_1, [1, 0]);  arg537_1 = None
        addmm_203: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg538_1, view_693, permute_385);  arg538_1 = view_693 = permute_385 = None
        view_694: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_203, [16, 128, 2560]);  addmm_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_697: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_694, [16, 128, -1, 80]);  view_694 = None
        permute_387: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_697, [0, 2, 1, 3]);  view_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_695: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_899, [2048, 2560]);  convert_element_type_899 = None
        permute_386: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg539_1, [1, 0]);  arg539_1 = None
        addmm_204: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg540_1, view_695, permute_386);  arg540_1 = view_695 = permute_386 = None
        view_696: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_204, [16, 128, 2560]);  addmm_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_698: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_696, [16, 128, -1, 80]);  view_696 = None
        permute_388: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_81: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_80: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_61: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_81, full_default_80);  full_default_81 = full_default_80 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_384, permute_387, permute_388, where_61, False, scale = 0.11180339887498948);  permute_384 = permute_387 = permute_388 = where_61 = None
        getitem_297: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_19[0];  _scaled_dot_product_cudnn_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_389: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_297, [0, 2, 1, 3]);  getitem_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_699: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_389, [16, 128, -1]);  permute_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_700: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_699, [2048, 2560]);  view_699 = None
        permute_390: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg541_1, [1, 0]);  arg541_1 = None
        addmm_205: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg542_1, view_700, permute_390);  arg542_1 = view_700 = permute_390 = None
        view_701: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_205, [16, 128, 2560]);  addmm_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_238: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_235, view_701);  add_235 = view_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_912: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_238, torch.float32)
        var_mean_63 = torch.ops.aten.var_mean.correction(convert_element_type_912, [2], correction = 0, keepdim = True)
        getitem_306: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_63[0]
        getitem_307: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_63[1];  var_mean_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_82: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_82 = None
        scalar_tensor_83: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_83 = None
        full_default_82: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_84: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_912, getitem_307);  convert_element_type_912 = getitem_307 = None
        add_239: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_306, 1e-05);  getitem_306 = None
        rsqrt_63: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_233: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_63);  sub_84 = rsqrt_63 = None
        mul_234: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, arg543_1);  mul_233 = arg543_1 = None
        add_240: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_234, arg544_1);  mul_234 = arg544_1 = None
        convert_element_type_913: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_240, torch.bfloat16);  add_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_702: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_913, [2048, 2560]);  convert_element_type_913 = None
        permute_391: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg545_1, [1, 0]);  arg545_1 = None
        addmm_206: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg546_1, view_702, permute_391);  arg546_1 = view_702 = permute_391 = None
        view_703: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_206, [16, 128, 2560]);  addmm_206 = None
        view_704: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_703, [16, 128, -1, 80]);  view_703 = None
        permute_392: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_235: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_392, 0.334370152488211);  permute_392 = None
        expand_87: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_235, [16, 32, 128, 80]);  mul_235 = None
        clone_169: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_711: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [512, 128, 80]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_705: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_393: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg547_1, [1, 0]);  arg547_1 = None
        addmm_207: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg548_1, view_705, permute_393);  arg548_1 = view_705 = permute_393 = None
        view_706: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_207, [16, 128, 2560]);  addmm_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_709: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_706, [16, 128, -1, 80]);  view_706 = None
        permute_395: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_709, [0, 2, 1, 3]);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_397: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_395, [0, 1, 3, 2]);  permute_395 = None
        mul_236: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_397, 0.334370152488211);  permute_397 = None
        expand_88: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_236, [16, 32, 80, 128]);  mul_236 = None
        clone_170: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_712: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [512, 80, 128]);  clone_170 = None
        bmm_42: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_711, view_712);  view_711 = view_712 = None
        view_713: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [16, 32, 128, 128]);  bmm_42 = None
        eq_21: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_713, -inf)
        logical_not_42: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_21);  eq_21 = None
        any_22: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_42, -1, True);  logical_not_42 = None
        logical_not_43: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        full_default_83: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_925: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_713, torch.float32);  view_713 = None
        amax_21: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_925, [-1], True)
        sub_85: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_925, amax_21);  convert_element_type_925 = amax_21 = None
        exp_21: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_85);  sub_85 = None
        sum_22: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_926: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        where_63: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_43, full_default_83, convert_element_type_926);  logical_not_43 = full_default_83 = convert_element_type_926 = None
        expand_89: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_63, [16, 32, 128, 128]);  where_63 = None
        view_714: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_89, [512, 128, 128]);  expand_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_707: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_394: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg549_1, [1, 0]);  arg549_1 = None
        addmm_208: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg550_1, view_707, permute_394);  arg550_1 = view_707 = permute_394 = None
        view_708: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_208, [16, 128, 2560]);  addmm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_710: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_708, [16, 128, -1, 80]);  view_708 = None
        permute_396: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_710, [0, 2, 1, 3]);  view_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_90: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_396, [16, 32, 128, 80]);  permute_396 = None
        clone_171: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_715: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [512, 128, 80]);  clone_171 = None
        bmm_43: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_714, view_715);  view_714 = view_715 = None
        view_716: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [16, 32, 128, 80]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_398: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_716, [0, 2, 1, 3]);  view_716 = None
        clone_172: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_398, memory_format = torch.contiguous_format);  permute_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_717: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [16, 128, -1]);  clone_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_718: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_717, [2048, 2560]);  view_717 = None
        permute_399: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg551_1, [1, 0]);  arg551_1 = None
        addmm_209: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg552_1, view_718, permute_399);  arg552_1 = view_718 = permute_399 = None
        view_719: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_209, [16, 128, 2560]);  addmm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_242: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_238, view_719);  add_238 = view_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_932: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_242, torch.float32)
        var_mean_64 = torch.ops.aten.var_mean.correction(convert_element_type_932, [2], correction = 0, keepdim = True)
        getitem_308: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_64[0]
        getitem_309: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_64[1];  var_mean_64 = None
        sub_86: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_932, getitem_309);  convert_element_type_932 = getitem_309 = None
        add_243: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_308, 1e-05);  getitem_308 = None
        rsqrt_64: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_237: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_64);  sub_86 = rsqrt_64 = None
        mul_238: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, arg553_1);  mul_237 = arg553_1 = None
        add_244: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_238, arg554_1);  mul_238 = arg554_1 = None
        convert_element_type_933: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16);  add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_720: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_933, [2048, 2560]);  convert_element_type_933 = None
        permute_400: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg555_1, [1, 0]);  arg555_1 = None
        addmm_210: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg556_1, view_720, permute_400);  arg556_1 = view_720 = permute_400 = None
        view_721: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_210, [16, 128, 10240]);  addmm_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_937: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_721, torch.float32);  view_721 = None
        mul_239: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_937, 0.5)
        mul_240: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_937, 0.7071067811865476);  convert_element_type_937 = None
        erf_21: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
        add_245: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_241: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_239, add_245);  mul_239 = add_245 = None
        convert_element_type_938: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_722: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_938, [2048, 10240]);  convert_element_type_938 = None
        permute_401: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg557_1, [1, 0]);  arg557_1 = None
        addmm_211: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg558_1, view_722, permute_401);  arg558_1 = view_722 = permute_401 = None
        view_723: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_211, [16, 128, 2560]);  addmm_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_246: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_242, view_723);  add_242 = view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_942: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_246, torch.float32)
        var_mean_65 = torch.ops.aten.var_mean.correction(convert_element_type_942, [2], correction = 0, keepdim = True)
        getitem_310: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_65[0]
        getitem_311: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_65[1];  var_mean_65 = None
        sub_87: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_942, getitem_311);  convert_element_type_942 = getitem_311 = None
        add_247: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_65: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_242: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_65);  sub_87 = rsqrt_65 = None
        mul_243: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, arg559_1);  mul_242 = arg559_1 = None
        add_248: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_243, arg560_1);  mul_243 = arg560_1 = None
        convert_element_type_943: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_248, torch.bfloat16);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_724: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_943, [2048, 2560])
        permute_402: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg561_1, [1, 0]);  arg561_1 = None
        addmm_212: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg562_1, view_724, permute_402);  arg562_1 = view_724 = permute_402 = None
        view_725: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_212, [16, 128, 2560]);  addmm_212 = None
        view_726: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_725, [16, 128, -1, 80]);  view_725 = None
        permute_403: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_726, [0, 2, 1, 3]);  view_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_727: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_943, [2048, 2560])
        permute_404: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg563_1, [1, 0]);  arg563_1 = None
        addmm_213: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg564_1, view_727, permute_404);  arg564_1 = view_727 = permute_404 = None
        view_728: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_213, [16, 128, 2560]);  addmm_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_731: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_728, [16, 128, -1, 80]);  view_728 = None
        permute_406: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_731, [0, 2, 1, 3]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_729: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_943, [2048, 2560]);  convert_element_type_943 = None
        permute_405: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg565_1, [1, 0]);  arg565_1 = None
        addmm_214: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg566_1, view_729, permute_405);  arg566_1 = view_729 = permute_405 = None
        view_730: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_214, [16, 128, 2560]);  addmm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_732: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_730, [16, 128, -1, 80]);  view_730 = None
        permute_407: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_85: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_84: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_64: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_85, full_default_84);  full_default_85 = full_default_84 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_403, permute_406, permute_407, where_64, False, scale = 0.11180339887498948);  permute_403 = permute_406 = permute_407 = where_64 = None
        getitem_312: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_20[0];  _scaled_dot_product_cudnn_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_408: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_312, [0, 2, 1, 3]);  getitem_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_733: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_408, [16, 128, -1]);  permute_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_734: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_733, [2048, 2560]);  view_733 = None
        permute_409: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg567_1, [1, 0]);  arg567_1 = None
        addmm_215: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg568_1, view_734, permute_409);  arg568_1 = view_734 = permute_409 = None
        view_735: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_215, [16, 128, 2560]);  addmm_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_249: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_246, view_735);  add_246 = view_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_956: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.float32)
        var_mean_66 = torch.ops.aten.var_mean.correction(convert_element_type_956, [2], correction = 0, keepdim = True)
        getitem_321: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_66[0]
        getitem_322: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_66[1];  var_mean_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_86: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_86 = None
        scalar_tensor_87: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_87 = None
        full_default_86: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_88: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_956, getitem_322);  convert_element_type_956 = getitem_322 = None
        add_250: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_321, 1e-05);  getitem_321 = None
        rsqrt_66: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        mul_244: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_66);  sub_88 = rsqrt_66 = None
        mul_245: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_244, arg569_1);  mul_244 = arg569_1 = None
        add_251: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_245, arg570_1);  mul_245 = arg570_1 = None
        convert_element_type_957: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_251, torch.bfloat16);  add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_736: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_957, [2048, 2560]);  convert_element_type_957 = None
        permute_410: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg571_1, [1, 0]);  arg571_1 = None
        addmm_216: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg572_1, view_736, permute_410);  arg572_1 = view_736 = permute_410 = None
        view_737: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_216, [16, 128, 2560]);  addmm_216 = None
        view_738: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_737, [16, 128, -1, 80]);  view_737 = None
        permute_411: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_246: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_411, 0.334370152488211);  permute_411 = None
        expand_91: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_246, [16, 32, 128, 80]);  mul_246 = None
        clone_177: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_745: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [512, 128, 80]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_739: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_412: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg573_1, [1, 0]);  arg573_1 = None
        addmm_217: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg574_1, view_739, permute_412);  arg574_1 = view_739 = permute_412 = None
        view_740: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_217, [16, 128, 2560]);  addmm_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_743: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_740, [16, 128, -1, 80]);  view_740 = None
        permute_414: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_743, [0, 2, 1, 3]);  view_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_416: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_414, [0, 1, 3, 2]);  permute_414 = None
        mul_247: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_416, 0.334370152488211);  permute_416 = None
        expand_92: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_247, [16, 32, 80, 128]);  mul_247 = None
        clone_178: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_746: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [512, 80, 128]);  clone_178 = None
        bmm_44: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_745, view_746);  view_745 = view_746 = None
        view_747: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [16, 32, 128, 128]);  bmm_44 = None
        eq_22: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_747, -inf)
        logical_not_44: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_22);  eq_22 = None
        any_23: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_44, -1, True);  logical_not_44 = None
        logical_not_45: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        full_default_87: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_969: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_747, torch.float32);  view_747 = None
        amax_22: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_969, [-1], True)
        sub_89: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_969, amax_22);  convert_element_type_969 = amax_22 = None
        exp_22: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_89);  sub_89 = None
        sum_23: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_970: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        where_66: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_45, full_default_87, convert_element_type_970);  logical_not_45 = full_default_87 = convert_element_type_970 = None
        expand_93: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_66, [16, 32, 128, 128]);  where_66 = None
        view_748: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_93, [512, 128, 128]);  expand_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_741: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_413: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg575_1, [1, 0]);  arg575_1 = None
        addmm_218: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg576_1, view_741, permute_413);  arg576_1 = view_741 = permute_413 = None
        view_742: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_218, [16, 128, 2560]);  addmm_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_744: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_742, [16, 128, -1, 80]);  view_742 = None
        permute_415: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_744, [0, 2, 1, 3]);  view_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_94: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_415, [16, 32, 128, 80]);  permute_415 = None
        clone_179: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_749: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [512, 128, 80]);  clone_179 = None
        bmm_45: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_748, view_749);  view_748 = view_749 = None
        view_750: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [16, 32, 128, 80]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_417: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None
        clone_180: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_751: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_180, [16, 128, -1]);  clone_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_752: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_751, [2048, 2560]);  view_751 = None
        permute_418: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg577_1, [1, 0]);  arg577_1 = None
        addmm_219: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg578_1, view_752, permute_418);  arg578_1 = view_752 = permute_418 = None
        view_753: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_219, [16, 128, 2560]);  addmm_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_253: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_249, view_753);  add_249 = view_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_976: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_253, torch.float32)
        var_mean_67 = torch.ops.aten.var_mean.correction(convert_element_type_976, [2], correction = 0, keepdim = True)
        getitem_323: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_67[0]
        getitem_324: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_67[1];  var_mean_67 = None
        sub_90: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_976, getitem_324);  convert_element_type_976 = getitem_324 = None
        add_254: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_323, 1e-05);  getitem_323 = None
        rsqrt_67: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        mul_248: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_67);  sub_90 = rsqrt_67 = None
        mul_249: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, arg579_1);  mul_248 = arg579_1 = None
        add_255: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_249, arg580_1);  mul_249 = arg580_1 = None
        convert_element_type_977: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_255, torch.bfloat16);  add_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_754: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_977, [2048, 2560]);  convert_element_type_977 = None
        permute_419: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg581_1, [1, 0]);  arg581_1 = None
        addmm_220: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg582_1, view_754, permute_419);  arg582_1 = view_754 = permute_419 = None
        view_755: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_220, [16, 128, 10240]);  addmm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_981: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_755, torch.float32);  view_755 = None
        mul_250: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_981, 0.5)
        mul_251: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_981, 0.7071067811865476);  convert_element_type_981 = None
        erf_22: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_251);  mul_251 = None
        add_256: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_252: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, add_256);  mul_250 = add_256 = None
        convert_element_type_982: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_756: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_982, [2048, 10240]);  convert_element_type_982 = None
        permute_420: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg583_1, [1, 0]);  arg583_1 = None
        addmm_221: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg584_1, view_756, permute_420);  arg584_1 = view_756 = permute_420 = None
        view_757: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_221, [16, 128, 2560]);  addmm_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_257: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_253, view_757);  add_253 = view_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_986: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_257, torch.float32)
        var_mean_68 = torch.ops.aten.var_mean.correction(convert_element_type_986, [2], correction = 0, keepdim = True)
        getitem_325: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_68[0]
        getitem_326: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_68[1];  var_mean_68 = None
        sub_91: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_986, getitem_326);  convert_element_type_986 = getitem_326 = None
        add_258: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_325, 1e-05);  getitem_325 = None
        rsqrt_68: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        mul_253: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_68);  sub_91 = rsqrt_68 = None
        mul_254: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, arg585_1);  mul_253 = arg585_1 = None
        add_259: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_254, arg586_1);  mul_254 = arg586_1 = None
        convert_element_type_987: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_758: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_987, [2048, 2560])
        permute_421: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg587_1, [1, 0]);  arg587_1 = None
        addmm_222: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg588_1, view_758, permute_421);  arg588_1 = view_758 = permute_421 = None
        view_759: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_222, [16, 128, 2560]);  addmm_222 = None
        view_760: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_759, [16, 128, -1, 80]);  view_759 = None
        permute_422: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_761: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_987, [2048, 2560])
        permute_423: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg589_1, [1, 0]);  arg589_1 = None
        addmm_223: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg590_1, view_761, permute_423);  arg590_1 = view_761 = permute_423 = None
        view_762: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_223, [16, 128, 2560]);  addmm_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_765: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_762, [16, 128, -1, 80]);  view_762 = None
        permute_425: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_765, [0, 2, 1, 3]);  view_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_763: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_987, [2048, 2560]);  convert_element_type_987 = None
        permute_424: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg591_1, [1, 0]);  arg591_1 = None
        addmm_224: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg592_1, view_763, permute_424);  arg592_1 = view_763 = permute_424 = None
        view_764: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_224, [16, 128, 2560]);  addmm_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_766: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_764, [16, 128, -1, 80]);  view_764 = None
        permute_426: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_89: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_88: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_67: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_89, full_default_88);  full_default_89 = full_default_88 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_422, permute_425, permute_426, where_67, False, scale = 0.11180339887498948);  permute_422 = permute_425 = permute_426 = where_67 = None
        getitem_327: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_21[0];  _scaled_dot_product_cudnn_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_427: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_327, [0, 2, 1, 3]);  getitem_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_767: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_427, [16, 128, -1]);  permute_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_768: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_767, [2048, 2560]);  view_767 = None
        permute_428: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg593_1, [1, 0]);  arg593_1 = None
        addmm_225: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg594_1, view_768, permute_428);  arg594_1 = view_768 = permute_428 = None
        view_769: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_225, [16, 128, 2560]);  addmm_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_260: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_257, view_769);  add_257 = view_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_1000: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_260, torch.float32)
        var_mean_69 = torch.ops.aten.var_mean.correction(convert_element_type_1000, [2], correction = 0, keepdim = True)
        getitem_336: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_69[0]
        getitem_337: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_69[1];  var_mean_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_90: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_90 = None
        scalar_tensor_91: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_91 = None
        full_default_90: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_92: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1000, getitem_337);  convert_element_type_1000 = getitem_337 = None
        add_261: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_336, 1e-05);  getitem_336 = None
        rsqrt_69: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        mul_255: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_69);  sub_92 = rsqrt_69 = None
        mul_256: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, arg595_1);  mul_255 = arg595_1 = None
        add_262: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, arg596_1);  mul_256 = arg596_1 = None
        convert_element_type_1001: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_262, torch.bfloat16);  add_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_770: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1001, [2048, 2560]);  convert_element_type_1001 = None
        permute_429: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg597_1, [1, 0]);  arg597_1 = None
        addmm_226: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg598_1, view_770, permute_429);  arg598_1 = view_770 = permute_429 = None
        view_771: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_226, [16, 128, 2560]);  addmm_226 = None
        view_772: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_771, [16, 128, -1, 80]);  view_771 = None
        permute_430: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_257: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_430, 0.334370152488211);  permute_430 = None
        expand_95: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_257, [16, 32, 128, 80]);  mul_257 = None
        clone_185: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_779: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [512, 128, 80]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_773: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_431: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg599_1, [1, 0]);  arg599_1 = None
        addmm_227: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg600_1, view_773, permute_431);  arg600_1 = view_773 = permute_431 = None
        view_774: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_227, [16, 128, 2560]);  addmm_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_777: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_774, [16, 128, -1, 80]);  view_774 = None
        permute_433: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_777, [0, 2, 1, 3]);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_435: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_433, [0, 1, 3, 2]);  permute_433 = None
        mul_258: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_435, 0.334370152488211);  permute_435 = None
        expand_96: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_258, [16, 32, 80, 128]);  mul_258 = None
        clone_186: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_780: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [512, 80, 128]);  clone_186 = None
        bmm_46: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_779, view_780);  view_779 = view_780 = None
        view_781: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [16, 32, 128, 128]);  bmm_46 = None
        eq_23: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_781, -inf)
        logical_not_46: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_23);  eq_23 = None
        any_24: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_46, -1, True);  logical_not_46 = None
        logical_not_47: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        full_default_91: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_1013: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_781, torch.float32);  view_781 = None
        amax_23: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_1013, [-1], True)
        sub_93: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1013, amax_23);  convert_element_type_1013 = amax_23 = None
        exp_23: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        sum_24: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_1014: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        where_69: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_47, full_default_91, convert_element_type_1014);  logical_not_47 = full_default_91 = convert_element_type_1014 = None
        expand_97: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_69, [16, 32, 128, 128]);  where_69 = None
        view_782: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_97, [512, 128, 128]);  expand_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_775: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_432: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg601_1, [1, 0]);  arg601_1 = None
        addmm_228: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg602_1, view_775, permute_432);  arg602_1 = view_775 = permute_432 = None
        view_776: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_228, [16, 128, 2560]);  addmm_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_778: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_776, [16, 128, -1, 80]);  view_776 = None
        permute_434: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_778, [0, 2, 1, 3]);  view_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_98: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_434, [16, 32, 128, 80]);  permute_434 = None
        clone_187: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_783: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_187, [512, 128, 80]);  clone_187 = None
        bmm_47: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_782, view_783);  view_782 = view_783 = None
        view_784: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [16, 32, 128, 80]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_436: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_784, [0, 2, 1, 3]);  view_784 = None
        clone_188: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_785: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_188, [16, 128, -1]);  clone_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_786: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_785, [2048, 2560]);  view_785 = None
        permute_437: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg603_1, [1, 0]);  arg603_1 = None
        addmm_229: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg604_1, view_786, permute_437);  arg604_1 = view_786 = permute_437 = None
        view_787: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_229, [16, 128, 2560]);  addmm_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_264: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_260, view_787);  add_260 = view_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_1020: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_264, torch.float32)
        var_mean_70 = torch.ops.aten.var_mean.correction(convert_element_type_1020, [2], correction = 0, keepdim = True)
        getitem_338: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_70[0]
        getitem_339: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_70[1];  var_mean_70 = None
        sub_94: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1020, getitem_339);  convert_element_type_1020 = getitem_339 = None
        add_265: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_338, 1e-05);  getitem_338 = None
        rsqrt_70: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
        mul_259: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_70);  sub_94 = rsqrt_70 = None
        mul_260: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, arg605_1);  mul_259 = arg605_1 = None
        add_266: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_260, arg606_1);  mul_260 = arg606_1 = None
        convert_element_type_1021: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_266, torch.bfloat16);  add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_788: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1021, [2048, 2560]);  convert_element_type_1021 = None
        permute_438: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg607_1, [1, 0]);  arg607_1 = None
        addmm_230: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg608_1, view_788, permute_438);  arg608_1 = view_788 = permute_438 = None
        view_789: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_230, [16, 128, 10240]);  addmm_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1025: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_789, torch.float32);  view_789 = None
        mul_261: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1025, 0.5)
        mul_262: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1025, 0.7071067811865476);  convert_element_type_1025 = None
        erf_23: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_262);  mul_262 = None
        add_267: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_263: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, add_267);  mul_261 = add_267 = None
        convert_element_type_1026: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_790: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1026, [2048, 10240]);  convert_element_type_1026 = None
        permute_439: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg609_1, [1, 0]);  arg609_1 = None
        addmm_231: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg610_1, view_790, permute_439);  arg610_1 = view_790 = permute_439 = None
        view_791: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_231, [16, 128, 2560]);  addmm_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_268: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_264, view_791);  add_264 = view_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_1030: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_268, torch.float32)
        var_mean_71 = torch.ops.aten.var_mean.correction(convert_element_type_1030, [2], correction = 0, keepdim = True)
        getitem_340: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_71[0]
        getitem_341: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_71[1];  var_mean_71 = None
        sub_95: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1030, getitem_341);  convert_element_type_1030 = getitem_341 = None
        add_269: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_340, 1e-05);  getitem_340 = None
        rsqrt_71: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
        mul_264: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_71);  sub_95 = rsqrt_71 = None
        mul_265: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_264, arg611_1);  mul_264 = arg611_1 = None
        add_270: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_265, arg612_1);  mul_265 = arg612_1 = None
        convert_element_type_1031: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_270, torch.bfloat16);  add_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_792: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1031, [2048, 2560])
        permute_440: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg613_1, [1, 0]);  arg613_1 = None
        addmm_232: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg614_1, view_792, permute_440);  arg614_1 = view_792 = permute_440 = None
        view_793: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_232, [16, 128, 2560]);  addmm_232 = None
        view_794: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_793, [16, 128, -1, 80]);  view_793 = None
        permute_441: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_794, [0, 2, 1, 3]);  view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_795: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1031, [2048, 2560])
        permute_442: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg615_1, [1, 0]);  arg615_1 = None
        addmm_233: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg616_1, view_795, permute_442);  arg616_1 = view_795 = permute_442 = None
        view_796: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_233, [16, 128, 2560]);  addmm_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_799: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_796, [16, 128, -1, 80]);  view_796 = None
        permute_444: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_799, [0, 2, 1, 3]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_797: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1031, [2048, 2560]);  convert_element_type_1031 = None
        permute_443: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg617_1, [1, 0]);  arg617_1 = None
        addmm_234: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg618_1, view_797, permute_443);  arg618_1 = view_797 = permute_443 = None
        view_798: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_234, [16, 128, 2560]);  addmm_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_800: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_798, [16, 128, -1, 80]);  view_798 = None
        permute_445: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_800, [0, 2, 1, 3]);  view_800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_93: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_92: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_70: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_93, full_default_92);  full_default_93 = full_default_92 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_441, permute_444, permute_445, where_70, False, scale = 0.11180339887498948);  permute_441 = permute_444 = permute_445 = where_70 = None
        getitem_342: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_22[0];  _scaled_dot_product_cudnn_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_446: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_342, [0, 2, 1, 3]);  getitem_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_801: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_446, [16, 128, -1]);  permute_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_802: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_801, [2048, 2560]);  view_801 = None
        permute_447: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg619_1, [1, 0]);  arg619_1 = None
        addmm_235: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg620_1, view_802, permute_447);  arg620_1 = view_802 = permute_447 = None
        view_803: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_235, [16, 128, 2560]);  addmm_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_271: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_268, view_803);  add_268 = view_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_1044: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_271, torch.float32)
        var_mean_72 = torch.ops.aten.var_mean.correction(convert_element_type_1044, [2], correction = 0, keepdim = True)
        getitem_351: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_72[0]
        getitem_352: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_72[1];  var_mean_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_94: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_94 = None
        scalar_tensor_95: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_95 = None
        full_default_94: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_96: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1044, getitem_352);  convert_element_type_1044 = getitem_352 = None
        add_272: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_351, 1e-05);  getitem_351 = None
        rsqrt_72: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        mul_266: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_72);  sub_96 = rsqrt_72 = None
        mul_267: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, arg621_1);  mul_266 = arg621_1 = None
        add_273: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_267, arg622_1);  mul_267 = arg622_1 = None
        convert_element_type_1045: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_273, torch.bfloat16);  add_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_804: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1045, [2048, 2560]);  convert_element_type_1045 = None
        permute_448: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg623_1, [1, 0]);  arg623_1 = None
        addmm_236: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg624_1, view_804, permute_448);  arg624_1 = view_804 = permute_448 = None
        view_805: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_236, [16, 128, 2560]);  addmm_236 = None
        view_806: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_805, [16, 128, -1, 80]);  view_805 = None
        permute_449: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_806, [0, 2, 1, 3]);  view_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_268: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_449, 0.334370152488211);  permute_449 = None
        expand_99: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_268, [16, 32, 128, 80]);  mul_268 = None
        clone_193: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_813: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [512, 128, 80]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_807: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_450: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg625_1, [1, 0]);  arg625_1 = None
        addmm_237: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg626_1, view_807, permute_450);  arg626_1 = view_807 = permute_450 = None
        view_808: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_237, [16, 128, 2560]);  addmm_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_811: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_808, [16, 128, -1, 80]);  view_808 = None
        permute_452: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_811, [0, 2, 1, 3]);  view_811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_454: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_452, [0, 1, 3, 2]);  permute_452 = None
        mul_269: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_454, 0.334370152488211);  permute_454 = None
        expand_100: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_269, [16, 32, 80, 128]);  mul_269 = None
        clone_194: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_814: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_194, [512, 80, 128]);  clone_194 = None
        bmm_48: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_813, view_814);  view_813 = view_814 = None
        view_815: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [16, 32, 128, 128]);  bmm_48 = None
        eq_24: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_815, -inf)
        logical_not_48: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_24);  eq_24 = None
        any_25: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_48, -1, True);  logical_not_48 = None
        logical_not_49: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_25);  any_25 = None
        full_default_95: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_1057: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_815, torch.float32);  view_815 = None
        amax_24: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_1057, [-1], True)
        sub_97: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1057, amax_24);  convert_element_type_1057 = amax_24 = None
        exp_24: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_97);  sub_97 = None
        sum_25: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_24: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        convert_element_type_1058: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None
        where_72: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_49, full_default_95, convert_element_type_1058);  logical_not_49 = full_default_95 = convert_element_type_1058 = None
        expand_101: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_72, [16, 32, 128, 128]);  where_72 = None
        view_816: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_101, [512, 128, 128]);  expand_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_809: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_451: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg627_1, [1, 0]);  arg627_1 = None
        addmm_238: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg628_1, view_809, permute_451);  arg628_1 = view_809 = permute_451 = None
        view_810: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_238, [16, 128, 2560]);  addmm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_812: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_810, [16, 128, -1, 80]);  view_810 = None
        permute_453: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_812, [0, 2, 1, 3]);  view_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_102: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_453, [16, 32, 128, 80]);  permute_453 = None
        clone_195: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_102, memory_format = torch.contiguous_format);  expand_102 = None
        view_817: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_195, [512, 128, 80]);  clone_195 = None
        bmm_49: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_816, view_817);  view_816 = view_817 = None
        view_818: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [16, 32, 128, 80]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_455: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_818, [0, 2, 1, 3]);  view_818 = None
        clone_196: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_819: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_196, [16, 128, -1]);  clone_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_820: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_819, [2048, 2560]);  view_819 = None
        permute_456: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg629_1, [1, 0]);  arg629_1 = None
        addmm_239: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg630_1, view_820, permute_456);  arg630_1 = view_820 = permute_456 = None
        view_821: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_239, [16, 128, 2560]);  addmm_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_275: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_271, view_821);  add_271 = view_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_1064: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_275, torch.float32)
        var_mean_73 = torch.ops.aten.var_mean.correction(convert_element_type_1064, [2], correction = 0, keepdim = True)
        getitem_353: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_73[0]
        getitem_354: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_73[1];  var_mean_73 = None
        sub_98: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1064, getitem_354);  convert_element_type_1064 = getitem_354 = None
        add_276: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_353, 1e-05);  getitem_353 = None
        rsqrt_73: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        mul_270: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_73);  sub_98 = rsqrt_73 = None
        mul_271: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, arg631_1);  mul_270 = arg631_1 = None
        add_277: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_271, arg632_1);  mul_271 = arg632_1 = None
        convert_element_type_1065: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_277, torch.bfloat16);  add_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_822: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1065, [2048, 2560]);  convert_element_type_1065 = None
        permute_457: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg633_1, [1, 0]);  arg633_1 = None
        addmm_240: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg634_1, view_822, permute_457);  arg634_1 = view_822 = permute_457 = None
        view_823: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_240, [16, 128, 10240]);  addmm_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1069: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_823, torch.float32);  view_823 = None
        mul_272: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1069, 0.5)
        mul_273: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1069, 0.7071067811865476);  convert_element_type_1069 = None
        erf_24: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_273);  mul_273 = None
        add_278: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_274: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_272, add_278);  mul_272 = add_278 = None
        convert_element_type_1070: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_274, torch.bfloat16);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_824: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1070, [2048, 10240]);  convert_element_type_1070 = None
        permute_458: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg635_1, [1, 0]);  arg635_1 = None
        addmm_241: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg636_1, view_824, permute_458);  arg636_1 = view_824 = permute_458 = None
        view_825: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_241, [16, 128, 2560]);  addmm_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_279: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_275, view_825);  add_275 = view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_1074: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.float32)
        var_mean_74 = torch.ops.aten.var_mean.correction(convert_element_type_1074, [2], correction = 0, keepdim = True)
        getitem_355: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_74[0]
        getitem_356: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_74[1];  var_mean_74 = None
        sub_99: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1074, getitem_356);  convert_element_type_1074 = getitem_356 = None
        add_280: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_355, 1e-05);  getitem_355 = None
        rsqrt_74: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        mul_275: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_74);  sub_99 = rsqrt_74 = None
        mul_276: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_275, arg637_1);  mul_275 = arg637_1 = None
        add_281: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_276, arg638_1);  mul_276 = arg638_1 = None
        convert_element_type_1075: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_281, torch.bfloat16);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_826: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1075, [2048, 2560])
        permute_459: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg639_1, [1, 0]);  arg639_1 = None
        addmm_242: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg640_1, view_826, permute_459);  arg640_1 = view_826 = permute_459 = None
        view_827: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_242, [16, 128, 2560]);  addmm_242 = None
        view_828: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_827, [16, 128, -1, 80]);  view_827 = None
        permute_460: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_828, [0, 2, 1, 3]);  view_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_829: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1075, [2048, 2560])
        permute_461: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg641_1, [1, 0]);  arg641_1 = None
        addmm_243: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg642_1, view_829, permute_461);  arg642_1 = view_829 = permute_461 = None
        view_830: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_243, [16, 128, 2560]);  addmm_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_833: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_830, [16, 128, -1, 80]);  view_830 = None
        permute_463: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_831: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1075, [2048, 2560]);  convert_element_type_1075 = None
        permute_462: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg643_1, [1, 0]);  arg643_1 = None
        addmm_244: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg644_1, view_831, permute_462);  arg644_1 = view_831 = permute_462 = None
        view_832: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_244, [16, 128, 2560]);  addmm_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_834: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_832, [16, 128, -1, 80]);  view_832 = None
        permute_464: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_834, [0, 2, 1, 3]);  view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_97: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_73: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_9, full_default_97, full_default_96);  expand_9 = full_default_97 = full_default_96 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_460, permute_463, permute_464, where_73, False, scale = 0.11180339887498948);  permute_460 = permute_463 = permute_464 = where_73 = None
        getitem_357: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_23[0];  _scaled_dot_product_cudnn_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_465: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_357, [0, 2, 1, 3]);  getitem_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_835: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_465, [16, 128, -1]);  permute_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_836: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_835, [2048, 2560]);  view_835 = None
        permute_466: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg645_1, [1, 0]);  arg645_1 = None
        addmm_245: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg646_1, view_836, permute_466);  arg646_1 = view_836 = permute_466 = None
        view_837: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_245, [16, 128, 2560]);  addmm_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_282: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_279, view_837);  add_279 = view_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_1088: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_282, torch.float32)
        var_mean_75 = torch.ops.aten.var_mean.correction(convert_element_type_1088, [2], correction = 0, keepdim = True)
        getitem_366: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_75[0]
        getitem_367: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_75[1];  var_mean_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_98: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_98 = None
        scalar_tensor_99: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_99 = None
        full_default_98: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_100: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1088, getitem_367);  convert_element_type_1088 = getitem_367 = None
        add_283: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_366, 1e-05);  getitem_366 = None
        rsqrt_75: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        mul_277: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_75);  sub_100 = rsqrt_75 = None
        mul_278: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_277, arg647_1);  mul_277 = arg647_1 = None
        add_284: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_278, arg648_1);  mul_278 = arg648_1 = None
        convert_element_type_1089: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_284, torch.bfloat16);  add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_838: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1089, [2048, 2560]);  convert_element_type_1089 = None
        permute_467: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg649_1, [1, 0]);  arg649_1 = None
        addmm_246: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg650_1, view_838, permute_467);  arg650_1 = view_838 = permute_467 = None
        view_839: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_246, [16, 128, 2560]);  addmm_246 = None
        view_840: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_839, [16, 128, -1, 80]);  view_839 = None
        permute_468: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_840, [0, 2, 1, 3]);  view_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_279: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_468, 0.334370152488211);  permute_468 = None
        expand_103: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(mul_279, [16, 32, 128, 80]);  mul_279 = None
        clone_201: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_847: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [512, 128, 80]);  clone_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_841: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_469: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg651_1, [1, 0]);  arg651_1 = None
        addmm_247: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg652_1, view_841, permute_469);  arg652_1 = view_841 = permute_469 = None
        view_842: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_247, [16, 128, 2560]);  addmm_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_845: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_842, [16, 128, -1, 80]);  view_842 = None
        permute_471: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_845, [0, 2, 1, 3]);  view_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_473: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.permute.default(permute_471, [0, 1, 3, 2]);  permute_471 = None
        mul_280: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.mul.Scalar(permute_473, 0.334370152488211);  permute_473 = None
        expand_104: "bf16[16, 32, 80, 128][327680, 80, 1, 2560]cuda:0" = torch.ops.aten.expand.default(mul_280, [16, 32, 80, 128]);  mul_280 = None
        clone_202: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_848: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_202, [512, 80, 128]);  clone_202 = None
        bmm_50: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_847, view_848);  view_847 = view_848 = None
        view_849: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [16, 32, 128, 128]);  bmm_50 = None
        eq_25: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_849, -inf)
        logical_not_50: "b8[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_25);  eq_25 = None
        any_26: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_50, -1, True);  logical_not_50 = None
        logical_not_51: "b8[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_26);  any_26 = None
        full_default_99: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_1101: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_849, torch.float32);  view_849 = None
        amax_25: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_1101, [-1], True)
        sub_101: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1101, amax_25);  convert_element_type_1101 = amax_25 = None
        exp_25: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_101);  sub_101 = None
        sum_26: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_25: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        convert_element_type_1102: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None
        where_75: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_51, full_default_99, convert_element_type_1102);  logical_not_51 = full_default_99 = convert_element_type_1102 = None
        expand_105: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_75, [16, 32, 128, 128]);  where_75 = None
        view_850: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_105, [512, 128, 128]);  expand_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_843: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [2048, 2560])
        permute_470: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg653_1, [1, 0]);  arg653_1 = None
        addmm_248: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg654_1, view_843, permute_470);  arg654_1 = view_843 = permute_470 = None
        view_844: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_248, [16, 128, 2560]);  addmm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_846: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_844, [16, 128, -1, 80]);  view_844 = None
        permute_472: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_106: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.expand.default(permute_472, [16, 32, 128, 80]);  permute_472 = None
        clone_203: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_851: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_203, [512, 128, 80]);  clone_203 = None
        bmm_51: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_850, view_851);  view_850 = view_851 = None
        view_852: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [16, 32, 128, 80]);  bmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_474: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_852, [0, 2, 1, 3]);  view_852 = None
        clone_204: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_474, memory_format = torch.contiguous_format);  permute_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_853: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_204, [16, 128, -1]);  clone_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_854: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_853, [2048, 2560]);  view_853 = None
        permute_475: "bf16[2560, 2560][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg655_1, [1, 0]);  arg655_1 = None
        addmm_249: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg656_1, view_854, permute_475);  arg656_1 = view_854 = permute_475 = None
        view_855: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_249, [16, 128, 2560]);  addmm_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_286: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_282, view_855);  add_282 = view_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_1108: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_286, torch.float32)
        var_mean_76 = torch.ops.aten.var_mean.correction(convert_element_type_1108, [2], correction = 0, keepdim = True)
        getitem_368: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_76[0]
        getitem_369: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_76[1];  var_mean_76 = None
        sub_102: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1108, getitem_369);  convert_element_type_1108 = getitem_369 = None
        add_287: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_368, 1e-05);  getitem_368 = None
        rsqrt_76: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_287);  add_287 = None
        mul_281: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_76);  sub_102 = rsqrt_76 = None
        mul_282: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, arg657_1);  mul_281 = arg657_1 = None
        add_288: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_282, arg658_1);  mul_282 = arg658_1 = None
        convert_element_type_1109: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_288, torch.bfloat16);  add_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_856: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1109, [2048, 2560]);  convert_element_type_1109 = None
        permute_476: "bf16[2560, 10240][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg659_1, [1, 0]);  arg659_1 = None
        addmm_250: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.addmm.default(arg660_1, view_856, permute_476);  arg660_1 = view_856 = permute_476 = None
        view_857: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_250, [16, 128, 10240]);  addmm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1113: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_857, torch.float32);  view_857 = None
        mul_283: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1113, 0.5)
        mul_284: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1113, 0.7071067811865476);  convert_element_type_1113 = None
        erf_25: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_284);  mul_284 = None
        add_289: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_285: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, add_289);  mul_283 = add_289 = None
        convert_element_type_1114: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_285, torch.bfloat16);  mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_858: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1114, [2048, 10240]);  convert_element_type_1114 = None
        permute_477: "bf16[10240, 2560][1, 10240]cuda:0" = torch.ops.aten.permute.default(arg661_1, [1, 0]);  arg661_1 = None
        addmm_251: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.addmm.default(arg662_1, view_858, permute_477);  arg662_1 = view_858 = permute_477 = None
        view_859: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_251, [16, 128, 2560]);  addmm_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_290: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_286, view_859);  add_286 = view_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:638 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_1118: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_290, torch.float32);  add_290 = None
        var_mean_77 = torch.ops.aten.var_mean.correction(convert_element_type_1118, [2], correction = 0, keepdim = True)
        getitem_370: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_77[0]
        getitem_371: "f32[16, 128, 1][128, 1, 1]cuda:0" = var_mean_77[1];  var_mean_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:892 in forward, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_863: "i64[2048][1]cuda:0" = torch.ops.aten.reshape.default(arg0_1, [-1]);  arg0_1 = None
        ne_1: "b8[2048][1]cuda:0" = torch.ops.aten.ne.Scalar(view_863, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:638 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_103: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1118, getitem_371);  convert_element_type_1118 = getitem_371 = None
        add_291: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_370, 1e-05);  getitem_370 = None
        rsqrt_77: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_291);  add_291 = None
        mul_286: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_77);  sub_103 = rsqrt_77 = None
        mul_287: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, arg663_1);  mul_286 = arg663_1 = None
        add_292: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_287, arg664_1);  mul_287 = arg664_1 = None
        convert_element_type_1119: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_292, torch.bfloat16);  add_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:885 in forward, code: lm_logits = self.lm_head(outputs[0])
        view_860: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1119, [2048, 2560]);  convert_element_type_1119 = None
        permute_478: "bf16[2560, 8008][1, 2560]cuda:0" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm: "bf16[2048, 8008][8008, 1]cuda:0" = torch.ops.aten.mm.default(view_860, permute_478);  view_860 = permute_478 = None
        view_861: "bf16[16, 128, 8008][1025024, 8008, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [16, 128, 8008]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:886 in forward, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        add_293: "bf16[16, 128, 8008][1025024, 8008, 1]cuda:0" = torch.ops.aten.add.Tensor(view_861, arg665_1);  view_861 = arg665_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:892 in forward, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_862: "bf16[2048, 8008][8008, 1]cuda:0" = torch.ops.aten.reshape.default(add_293, [-1, 8008])
        convert_element_type_1122: "f32[2048, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_862, torch.float32);  view_862 = None
        amax_26: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_1122, [1], True)
        sub_104: "f32[2048, 8008][8008, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1122, amax_26);  convert_element_type_1122 = amax_26 = None
        exp_26: "f32[2048, 8008][8008, 1]cuda:0" = torch.ops.aten.exp.default(sub_104)
        sum_27: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_26, [1], True);  exp_26 = None
        log: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_27);  sum_27 = None
        sub_105: "f32[2048, 8008][8008, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_104, log);  sub_104 = log = None
        convert_element_type_1123: "bf16[2048, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_105, torch.bfloat16);  sub_105 = None
        ne: "b8[2048][1]cuda:0" = torch.ops.aten.ne.Scalar(view_863, -100)
        full_default_100: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_76: "i64[2048][1]cuda:0" = torch.ops.aten.where.self(ne, view_863, full_default_100);  ne = full_default_100 = None
        unsqueeze_12: "i64[2048, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_76, 1);  where_76 = None
        gather: "bf16[2048, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_1123, 1, unsqueeze_12);  convert_element_type_1123 = unsqueeze_12 = None
        squeeze: "bf16[2048][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[2048][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_101: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_77: "bf16[2048][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_101);  ne_1 = neg = full_default_101 = None
        sum_29: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_77);  where_77 = None
        ne_2: "b8[2048][1]cuda:0" = torch.ops.aten.ne.Scalar(view_863, -100);  view_863 = None
        sum_28: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1124: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_28, torch.bfloat16);  sum_28 = None
        div_26: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_29, convert_element_type_1124);  sum_29 = convert_element_type_1124 = None
        return (div_26, add_293, convert_element_type_61)

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 512][512, 1]cuda:0", arg1_1: "bf16[32000, 1024][1024, 1]cuda:0", arg2_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg3_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg4_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg5_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg6_1: "bf16[16, 64][64, 1]cuda:0", arg7_1: "bf16[16, 64][64, 1]cuda:0", arg8_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg9_1: "bf16[1024][1]cuda:0", arg10_1: "bf16[1024][1]cuda:0", arg11_1: "bf16[4096, 1024][1024, 1]cuda:0", arg12_1: "bf16[4096][1]cuda:0", arg13_1: "bf16[1024, 4096][4096, 1]cuda:0", arg14_1: "bf16[1024][1]cuda:0", arg15_1: "bf16[1024][1]cuda:0", arg16_1: "bf16[1024][1]cuda:0", arg17_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg18_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg19_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg20_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg21_1: "bf16[16, 64][64, 1]cuda:0", arg22_1: "bf16[16, 64][64, 1]cuda:0", arg23_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg24_1: "bf16[1024][1]cuda:0", arg25_1: "bf16[1024][1]cuda:0", arg26_1: "bf16[4096, 1024][1024, 1]cuda:0", arg27_1: "bf16[4096][1]cuda:0", arg28_1: "bf16[1024, 4096][4096, 1]cuda:0", arg29_1: "bf16[1024][1]cuda:0", arg30_1: "bf16[1024][1]cuda:0", arg31_1: "bf16[1024][1]cuda:0", arg32_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg33_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg34_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg35_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg36_1: "bf16[16, 64][64, 1]cuda:0", arg37_1: "bf16[16, 64][64, 1]cuda:0", arg38_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg39_1: "bf16[1024][1]cuda:0", arg40_1: "bf16[1024][1]cuda:0", arg41_1: "bf16[4096, 1024][1024, 1]cuda:0", arg42_1: "bf16[4096][1]cuda:0", arg43_1: "bf16[1024, 4096][4096, 1]cuda:0", arg44_1: "bf16[1024][1]cuda:0", arg45_1: "bf16[1024][1]cuda:0", arg46_1: "bf16[1024][1]cuda:0", arg47_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg48_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg49_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg50_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg51_1: "bf16[16, 64][64, 1]cuda:0", arg52_1: "bf16[16, 64][64, 1]cuda:0", arg53_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg54_1: "bf16[1024][1]cuda:0", arg55_1: "bf16[1024][1]cuda:0", arg56_1: "bf16[4096, 1024][1024, 1]cuda:0", arg57_1: "bf16[4096][1]cuda:0", arg58_1: "bf16[1024, 4096][4096, 1]cuda:0", arg59_1: "bf16[1024][1]cuda:0", arg60_1: "bf16[1024][1]cuda:0", arg61_1: "bf16[1024][1]cuda:0", arg62_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg63_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg64_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg65_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg66_1: "bf16[16, 64][64, 1]cuda:0", arg67_1: "bf16[16, 64][64, 1]cuda:0", arg68_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg69_1: "bf16[1024][1]cuda:0", arg70_1: "bf16[1024][1]cuda:0", arg71_1: "bf16[4096, 1024][1024, 1]cuda:0", arg72_1: "bf16[4096][1]cuda:0", arg73_1: "bf16[1024, 4096][4096, 1]cuda:0", arg74_1: "bf16[1024][1]cuda:0", arg75_1: "bf16[1024][1]cuda:0", arg76_1: "bf16[1024][1]cuda:0", arg77_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg78_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg79_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg80_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg81_1: "bf16[16, 64][64, 1]cuda:0", arg82_1: "bf16[16, 64][64, 1]cuda:0", arg83_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg84_1: "bf16[1024][1]cuda:0", arg85_1: "bf16[1024][1]cuda:0", arg86_1: "bf16[4096, 1024][1024, 1]cuda:0", arg87_1: "bf16[4096][1]cuda:0", arg88_1: "bf16[1024, 4096][4096, 1]cuda:0", arg89_1: "bf16[1024][1]cuda:0", arg90_1: "bf16[1024][1]cuda:0", arg91_1: "bf16[1024][1]cuda:0", arg92_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg93_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg94_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg95_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg96_1: "bf16[16, 64][64, 1]cuda:0", arg97_1: "bf16[16, 64][64, 1]cuda:0", arg98_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg99_1: "bf16[1024][1]cuda:0", arg100_1: "bf16[1024][1]cuda:0", arg101_1: "bf16[4096, 1024][1024, 1]cuda:0", arg102_1: "bf16[4096][1]cuda:0", arg103_1: "bf16[1024, 4096][4096, 1]cuda:0", arg104_1: "bf16[1024][1]cuda:0", arg105_1: "bf16[1024][1]cuda:0", arg106_1: "bf16[1024][1]cuda:0", arg107_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg108_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg109_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg110_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg111_1: "bf16[16, 64][64, 1]cuda:0", arg112_1: "bf16[16, 64][64, 1]cuda:0", arg113_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg114_1: "bf16[1024][1]cuda:0", arg115_1: "bf16[1024][1]cuda:0", arg116_1: "bf16[4096, 1024][1024, 1]cuda:0", arg117_1: "bf16[4096][1]cuda:0", arg118_1: "bf16[1024, 4096][4096, 1]cuda:0", arg119_1: "bf16[1024][1]cuda:0", arg120_1: "bf16[1024][1]cuda:0", arg121_1: "bf16[1024][1]cuda:0", arg122_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg123_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg124_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg125_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg126_1: "bf16[16, 64][64, 1]cuda:0", arg127_1: "bf16[16, 64][64, 1]cuda:0", arg128_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg129_1: "bf16[1024][1]cuda:0", arg130_1: "bf16[1024][1]cuda:0", arg131_1: "bf16[4096, 1024][1024, 1]cuda:0", arg132_1: "bf16[4096][1]cuda:0", arg133_1: "bf16[1024, 4096][4096, 1]cuda:0", arg134_1: "bf16[1024][1]cuda:0", arg135_1: "bf16[1024][1]cuda:0", arg136_1: "bf16[1024][1]cuda:0", arg137_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg138_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg139_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg140_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg141_1: "bf16[16, 64][64, 1]cuda:0", arg142_1: "bf16[16, 64][64, 1]cuda:0", arg143_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg144_1: "bf16[1024][1]cuda:0", arg145_1: "bf16[1024][1]cuda:0", arg146_1: "bf16[4096, 1024][1024, 1]cuda:0", arg147_1: "bf16[4096][1]cuda:0", arg148_1: "bf16[1024, 4096][4096, 1]cuda:0", arg149_1: "bf16[1024][1]cuda:0", arg150_1: "bf16[1024][1]cuda:0", arg151_1: "bf16[1024][1]cuda:0", arg152_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg153_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg154_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg155_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg156_1: "bf16[16, 64][64, 1]cuda:0", arg157_1: "bf16[16, 64][64, 1]cuda:0", arg158_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg159_1: "bf16[1024][1]cuda:0", arg160_1: "bf16[1024][1]cuda:0", arg161_1: "bf16[4096, 1024][1024, 1]cuda:0", arg162_1: "bf16[4096][1]cuda:0", arg163_1: "bf16[1024, 4096][4096, 1]cuda:0", arg164_1: "bf16[1024][1]cuda:0", arg165_1: "bf16[1024][1]cuda:0", arg166_1: "bf16[1024][1]cuda:0", arg167_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg168_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg169_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg170_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg171_1: "bf16[16, 64][64, 1]cuda:0", arg172_1: "bf16[16, 64][64, 1]cuda:0", arg173_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg174_1: "bf16[1024][1]cuda:0", arg175_1: "bf16[1024][1]cuda:0", arg176_1: "bf16[4096, 1024][1024, 1]cuda:0", arg177_1: "bf16[4096][1]cuda:0", arg178_1: "bf16[1024, 4096][4096, 1]cuda:0", arg179_1: "bf16[1024][1]cuda:0", arg180_1: "bf16[1024][1]cuda:0", arg181_1: "bf16[1024][1]cuda:0", arg182_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg183_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg184_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg185_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg186_1: "bf16[16, 64][64, 1]cuda:0", arg187_1: "bf16[16, 64][64, 1]cuda:0", arg188_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg189_1: "bf16[1024][1]cuda:0", arg190_1: "bf16[1024][1]cuda:0", arg191_1: "bf16[4096, 1024][1024, 1]cuda:0", arg192_1: "bf16[4096][1]cuda:0", arg193_1: "bf16[1024, 4096][4096, 1]cuda:0", arg194_1: "bf16[1024][1]cuda:0", arg195_1: "bf16[1024][1]cuda:0", arg196_1: "bf16[1024][1]cuda:0", arg197_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg198_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg199_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg200_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg201_1: "bf16[16, 64][64, 1]cuda:0", arg202_1: "bf16[16, 64][64, 1]cuda:0", arg203_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg204_1: "bf16[1024][1]cuda:0", arg205_1: "bf16[1024][1]cuda:0", arg206_1: "bf16[4096, 1024][1024, 1]cuda:0", arg207_1: "bf16[4096][1]cuda:0", arg208_1: "bf16[1024, 4096][4096, 1]cuda:0", arg209_1: "bf16[1024][1]cuda:0", arg210_1: "bf16[1024][1]cuda:0", arg211_1: "bf16[1024][1]cuda:0", arg212_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg213_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg214_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg215_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg216_1: "bf16[16, 64][64, 1]cuda:0", arg217_1: "bf16[16, 64][64, 1]cuda:0", arg218_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg219_1: "bf16[1024][1]cuda:0", arg220_1: "bf16[1024][1]cuda:0", arg221_1: "bf16[4096, 1024][1024, 1]cuda:0", arg222_1: "bf16[4096][1]cuda:0", arg223_1: "bf16[1024, 4096][4096, 1]cuda:0", arg224_1: "bf16[1024][1]cuda:0", arg225_1: "bf16[1024][1]cuda:0", arg226_1: "bf16[1024][1]cuda:0", arg227_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg228_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg229_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg230_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg231_1: "bf16[16, 64][64, 1]cuda:0", arg232_1: "bf16[16, 64][64, 1]cuda:0", arg233_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg234_1: "bf16[1024][1]cuda:0", arg235_1: "bf16[1024][1]cuda:0", arg236_1: "bf16[4096, 1024][1024, 1]cuda:0", arg237_1: "bf16[4096][1]cuda:0", arg238_1: "bf16[1024, 4096][4096, 1]cuda:0", arg239_1: "bf16[1024][1]cuda:0", arg240_1: "bf16[1024][1]cuda:0", arg241_1: "bf16[1024][1]cuda:0", arg242_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg243_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg244_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg245_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg246_1: "bf16[16, 64][64, 1]cuda:0", arg247_1: "bf16[16, 64][64, 1]cuda:0", arg248_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg249_1: "bf16[1024][1]cuda:0", arg250_1: "bf16[1024][1]cuda:0", arg251_1: "bf16[4096, 1024][1024, 1]cuda:0", arg252_1: "bf16[4096][1]cuda:0", arg253_1: "bf16[1024, 4096][4096, 1]cuda:0", arg254_1: "bf16[1024][1]cuda:0", arg255_1: "bf16[1024][1]cuda:0", arg256_1: "bf16[1024][1]cuda:0", arg257_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg258_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg259_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg260_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg261_1: "bf16[16, 64][64, 1]cuda:0", arg262_1: "bf16[16, 64][64, 1]cuda:0", arg263_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg264_1: "bf16[1024][1]cuda:0", arg265_1: "bf16[1024][1]cuda:0", arg266_1: "bf16[4096, 1024][1024, 1]cuda:0", arg267_1: "bf16[4096][1]cuda:0", arg268_1: "bf16[1024, 4096][4096, 1]cuda:0", arg269_1: "bf16[1024][1]cuda:0", arg270_1: "bf16[1024][1]cuda:0", arg271_1: "bf16[1024][1]cuda:0", arg272_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg273_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg274_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg275_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg276_1: "bf16[16, 64][64, 1]cuda:0", arg277_1: "bf16[16, 64][64, 1]cuda:0", arg278_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg279_1: "bf16[1024][1]cuda:0", arg280_1: "bf16[1024][1]cuda:0", arg281_1: "bf16[4096, 1024][1024, 1]cuda:0", arg282_1: "bf16[4096][1]cuda:0", arg283_1: "bf16[1024, 4096][4096, 1]cuda:0", arg284_1: "bf16[1024][1]cuda:0", arg285_1: "bf16[1024][1]cuda:0", arg286_1: "bf16[1024][1]cuda:0", arg287_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg288_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg289_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg290_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg291_1: "bf16[16, 64][64, 1]cuda:0", arg292_1: "bf16[16, 64][64, 1]cuda:0", arg293_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg294_1: "bf16[1024][1]cuda:0", arg295_1: "bf16[1024][1]cuda:0", arg296_1: "bf16[4096, 1024][1024, 1]cuda:0", arg297_1: "bf16[4096][1]cuda:0", arg298_1: "bf16[1024, 4096][4096, 1]cuda:0", arg299_1: "bf16[1024][1]cuda:0", arg300_1: "bf16[1024][1]cuda:0", arg301_1: "bf16[1024][1]cuda:0", arg302_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg303_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg304_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg305_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg306_1: "bf16[16, 64][64, 1]cuda:0", arg307_1: "bf16[16, 64][64, 1]cuda:0", arg308_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg309_1: "bf16[1024][1]cuda:0", arg310_1: "bf16[1024][1]cuda:0", arg311_1: "bf16[4096, 1024][1024, 1]cuda:0", arg312_1: "bf16[4096][1]cuda:0", arg313_1: "bf16[1024, 4096][4096, 1]cuda:0", arg314_1: "bf16[1024][1]cuda:0", arg315_1: "bf16[1024][1]cuda:0", arg316_1: "bf16[1024][1]cuda:0", arg317_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg318_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg319_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg320_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg321_1: "bf16[16, 64][64, 1]cuda:0", arg322_1: "bf16[16, 64][64, 1]cuda:0", arg323_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg324_1: "bf16[1024][1]cuda:0", arg325_1: "bf16[1024][1]cuda:0", arg326_1: "bf16[4096, 1024][1024, 1]cuda:0", arg327_1: "bf16[4096][1]cuda:0", arg328_1: "bf16[1024, 4096][4096, 1]cuda:0", arg329_1: "bf16[1024][1]cuda:0", arg330_1: "bf16[1024][1]cuda:0", arg331_1: "bf16[1024][1]cuda:0", arg332_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg333_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg334_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg335_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg336_1: "bf16[16, 64][64, 1]cuda:0", arg337_1: "bf16[16, 64][64, 1]cuda:0", arg338_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg339_1: "bf16[1024][1]cuda:0", arg340_1: "bf16[1024][1]cuda:0", arg341_1: "bf16[4096, 1024][1024, 1]cuda:0", arg342_1: "bf16[4096][1]cuda:0", arg343_1: "bf16[1024, 4096][4096, 1]cuda:0", arg344_1: "bf16[1024][1]cuda:0", arg345_1: "bf16[1024][1]cuda:0", arg346_1: "bf16[1024][1]cuda:0", arg347_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg348_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg349_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg350_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg351_1: "bf16[16, 64][64, 1]cuda:0", arg352_1: "bf16[16, 64][64, 1]cuda:0", arg353_1: "bf16[1024, 16, 64][1024, 64, 1]cuda:0", arg354_1: "bf16[1024][1]cuda:0", arg355_1: "bf16[1024][1]cuda:0", arg356_1: "bf16[4096, 1024][1024, 1]cuda:0", arg357_1: "bf16[4096][1]cuda:0", arg358_1: "bf16[1024, 4096][4096, 1]cuda:0", arg359_1: "bf16[1024][1]cuda:0", arg360_1: "bf16[1024][1]cuda:0", arg361_1: "bf16[1024][1]cuda:0", arg362_1: "bf16[32000][1]cuda:0", arg363_1: "i64[16, 512][512, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1046 in forward, code: input_ids = input_ids.transpose(0, 1).contiguous()
        permute: "i64[512, 16][1, 512]cuda:0" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16][16, 1]cuda:0" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1115 in forward, code: word_emb_k = self.word_embedding(input_ids)
        embedding: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, clone);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_3: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_4: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
        view: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_4, [1, 8192, 1024]);  unsqueeze_4 = None
        squeeze_dim_238: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        unsqueeze_5: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg2_1, 3);  arg2_1 = None
        unsqueeze_6: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
        view_1: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_6, [1, 1024, 1024]);  unsqueeze_6 = None
        squeeze_dim_239: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1, 0);  view_1 = None
        mm_default_119: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_238, squeeze_dim_239);  squeeze_dim_238 = squeeze_dim_239 = None
        unsqueeze_default_119: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_119, 0);  mm_default_119 = None
        view_2: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_119, [512, 16, 1, 16, 64]);  unsqueeze_default_119 = None
        permute_7: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_7, [512, 16, 16, 64]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_3, arg6_1);  arg6_1 = None
        unsqueeze_19: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 4);  add = None
        permute_23: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_19, [1, 2, 0, 4, 3]);  unsqueeze_19 = None
        permute_25: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [0, 1, 2, 4, 3]);  permute_23 = None
        view_16: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_25, [256, 512, 64]);  permute_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_7: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_8: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 4);  unsqueeze_7 = None
        view_4: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_8, [1, 8192, 1024]);  unsqueeze_8 = None
        squeeze_dim_236: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_4, 0);  view_4 = None
        unsqueeze_9: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg3_1, 3);  arg3_1 = None
        unsqueeze_10: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 4);  unsqueeze_9 = None
        view_5: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_10, [1, 1024, 1024]);  unsqueeze_10 = None
        squeeze_dim_237: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_5, 0);  view_5 = None
        mm_default_118: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_236, squeeze_dim_237);  squeeze_dim_236 = squeeze_dim_237 = None
        unsqueeze_default_118: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_118, 0);  mm_default_118 = None
        view_6: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_118, [512, 16, 1, 16, 64]);  unsqueeze_default_118 = None
        permute_12: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 4, 2]);  view_6 = None
        view_7: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_12, [512, 16, 16, 64]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_20: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_7, 4);  view_7 = None
        permute_24: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_20, [1, 2, 4, 0, 3]);  unsqueeze_20 = None
        permute_26: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [0, 1, 4, 3, 2]);  permute_24 = None
        view_17: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_26, [256, 64, 512]);  permute_26 = None
        bmm_4: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = view_17 = None
        view_18: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [16, 16, 512, 1, 512]);  bmm_4 = None
        permute_27: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 1, 2, 4, 3]);  view_18 = None
        view_19: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [16, 16, 512, 512]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_1: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_3, arg7_1);  view_3 = arg7_1 = None
        unsqueeze_21: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 4);  add_1 = None
        permute_28: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_21, [1, 2, 0, 4, 3]);  unsqueeze_21 = None
        permute_30: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_28, [0, 1, 2, 4, 3]);  permute_28 = None
        view_20: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_30, [256, 512, 64]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:971 in relative_positional_encoding, code: fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.int64, device=device).float()
        iota_1: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_1: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:931 in positional_embedding, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        unsqueeze: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None
        permute_1: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:942 in relative_positional_encoding, code: freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.int64, device=device).float()
        iota: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:943 in relative_positional_encoding, code: inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))
        div: "f32[512][1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type, 1024);  convert_element_type = None
        pow_1: "f32[512][1]cuda:0" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512][1]cuda:0" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:931 in positional_embedding, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        unsqueeze_1: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul, 1);  mul = None
        permute_2: "f32[1, 512][1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_1: "f32[1024, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1, permute_2);  permute_1 = permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:932 in positional_embedding, code: pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        sin: "f32[1024, 512][512, 1]cuda:0" = torch.ops.aten.sin.default(mul_1)
        cos: "f32[1024, 512][512, 1]cuda:0" = torch.ops.aten.cos.default(mul_1);  mul_1 = None
        cat: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:933 in positional_embedding, code: pos_emb = pos_emb[:, None, :]
        unsqueeze_2: "f32[1024, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:936 in positional_embedding, code: pos_emb = pos_emb.expand(-1, bsz, -1)
        expand: "f32[1024, 16, 1024][1024, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_2, [-1, 16, -1]);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1143 in forward, code: pos_emb = self.dropout(pos_emb)
        clone_2: "f32[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_8: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_15: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_8, 3);  convert_element_type_8 = None
        unsqueeze_16: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 4);  unsqueeze_15 = None
        view_12: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_16, [1, 16384, 1024]);  unsqueeze_16 = None
        squeeze_dim_232: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_12, 0);  view_12 = None
        unsqueeze_17: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg5_1, 3);  arg5_1 = None
        unsqueeze_18: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
        view_13: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_18, [1, 1024, 1024]);  unsqueeze_18 = None
        squeeze_dim_233: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_13, 0);  view_13 = None
        mm_default_116: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_232, squeeze_dim_233);  squeeze_dim_232 = squeeze_dim_233 = None
        unsqueeze_default_116: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_116, 0);  mm_default_116 = None
        view_14: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_116, [1024, 16, 1, 16, 64]);  unsqueeze_default_116 = None
        permute_22: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 1, 3, 4, 2]);  view_14 = None
        view_15: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_22, [1024, 16, 16, 64]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_22: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_15, 4);  view_15 = None
        permute_29: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_22, [1, 2, 4, 0, 3]);  unsqueeze_22 = None
        permute_31: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [0, 1, 4, 3, 2]);  permute_29 = None
        view_21: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_31, [256, 64, 1024]);  permute_31 = None
        bmm_5: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_20, view_21);  view_20 = view_21 = None
        view_22: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [16, 16, 512, 1, 1024]);  bmm_5 = None
        permute_32: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_22, [0, 1, 2, 4, 3]);  view_22 = None
        view_23: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_32, [16, 16, 512, 1024]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_24: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [16, 16, 1024, 512]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_2: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_24, 2, 1, 9223372036854775807);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_25: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_2, [16, 16, 512, 1023]);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_2: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_25, [None, None, None, iota_2]);  view_25 = iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_2: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_19, index);  view_19 = index = None
        add_3: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_2, 0);  add_2 = None

        # No stacktrace found for following nodes
        mul_tensor_92: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_3, 0.125)
        convert_element_type_default_46: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_92, torch.float32);  mul_tensor_92 = None
        eq_tensor_23: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_46, convert_element_type_default_46)
        abs_default_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_46)
        ne_scalar_23: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_23, inf);  abs_default_23 = None
        mul_tensor_95: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_23, ne_scalar_23);  eq_tensor_23 = ne_scalar_23 = None
        logical_not_default_46: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_95);  mul_tensor_95 = None
        any_dims_23: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_46, [3], True);  logical_not_default_46 = None
        logical_not_default_47: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_23);  any_dims_23 = None
        convert_element_type_default_47: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        mul_tensor_93: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_47, 1);  convert_element_type_default_47 = None
        amax_default_46: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_93, [3], True)
        sub_tensor_46: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_93, amax_default_46);  mul_tensor_93 = amax_default_46 = None
        mul_tensor_94: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_46, 0.125);  sub_tensor_46 = None
        amax_default_47: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_46, [3], True)
        sub_tensor_47: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_46, amax_default_47);  convert_element_type_default_46 = amax_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_47, mul_tensor_94, sub_tensor_47);  logical_not_default_47 = mul_tensor_94 = sub_tensor_47 = None
        exp: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_23);  where_self_23 = None
        sum_1: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_16: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_23: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_16, 4);  convert_element_type_16 = None
        view_26: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_23, [256, 512, 512]);  unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_11: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_12: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
        view_8: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_12, [1, 8192, 1024]);  unsqueeze_12 = None
        squeeze_dim_234: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_8, 0);  view_8 = None
        unsqueeze_13: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg4_1, 3);  arg4_1 = None
        unsqueeze_14: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 4);  unsqueeze_13 = None
        view_9: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_14, [1, 1024, 1024]);  unsqueeze_14 = None
        squeeze_dim_235: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_9, 0);  view_9 = None
        mm_default_117: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_234, squeeze_dim_235);  squeeze_dim_234 = squeeze_dim_235 = None
        unsqueeze_default_117: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_117, 0);  mm_default_117 = None
        view_10: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_117, [512, 16, 1, 16, 64]);  unsqueeze_default_117 = None
        permute_17: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 1, 3, 4, 2]);  view_10 = None
        view_11: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_17, [512, 16, 16, 64]);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_24: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_11, 4);  view_11 = None
        permute_34: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_24, [4, 1, 2, 3, 0]);  unsqueeze_24 = None
        permute_36: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 2, 4, 3, 0]);  permute_34 = None
        view_27: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_36, [256, 512, 64]);  permute_36 = None
        bmm_6: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_26, view_27);  view_26 = view_27 = None
        view_28: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [16, 16, 512, 1, 64]);  bmm_6 = None
        permute_37: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_28, [2, 0, 1, 4, 3]);  view_28 = None
        view_29: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_37, [512, 16, 16, 64]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_25: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_29, 4);  view_29 = None
        permute_38: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_25, [0, 1, 4, 3, 2]);  unsqueeze_25 = None
        permute_40: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 4, 2]);  permute_38 = None
        clone_4: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_30: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [1, 8192, 1024]);  clone_4 = None
        squeeze_dim_230: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_30, 0);  view_30 = None
        unsqueeze_26: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 3);  arg8_1 = None
        unsqueeze_27: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 4);  unsqueeze_26 = None
        permute_39: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_27, [3, 4, 0, 2, 1]);  unsqueeze_27 = None
        permute_41: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_39, [3, 4, 2, 0, 1]);  permute_39 = None
        clone_5: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_31: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [1, 1024, 1024]);  clone_5 = None
        squeeze_dim_231: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_31, 0);  view_31 = None
        mm_default_115: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_230, squeeze_dim_231);  squeeze_dim_230 = squeeze_dim_231 = None
        unsqueeze_default_115: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_115, 0);  mm_default_115 = None
        view_32: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_115, [512, 16, 1, 1, 1024]);  unsqueeze_default_115 = None
        permute_42: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 1, 4, 2, 3]);  view_32 = None
        view_33: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_42, [512, 16, 1024]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_4: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_33, embedding);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_21: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_21, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub_1: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_21, getitem_1);  convert_element_type_21 = getitem_1 = None
        add_5: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_3: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_4: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, arg9_1);  mul_3 = arg9_1 = None
        add_6: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, arg10_1);  mul_4 = arg10_1 = None
        convert_element_type_22: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_34: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_22, [8192, 1024])
        permute_43: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view_34, permute_43);  arg12_1 = view_34 = permute_43 = None
        view_35: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [512, 16, 4096]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_26: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        mul_5: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.5)
        mul_6: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476);  convert_element_type_26 = None
        erf: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_7: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, add_7);  mul_5 = add_7 = None
        convert_element_type_27: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_36: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_27, [8192, 4096]);  convert_element_type_27 = None
        permute_44: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg14_1, view_36, permute_44);  arg14_1 = view_36 = permute_44 = None
        view_37: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [512, 16, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_8: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_37, convert_element_type_22);  view_37 = convert_element_type_22 = None
        convert_element_type_31: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_31, [2], correction = 0, keepdim = True)
        getitem_2: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_31, getitem_3);  convert_element_type_31 = getitem_3 = None
        add_9: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_8: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_9: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_10: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None
        convert_element_type_32: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_28: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_32, 3)
        unsqueeze_29: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 4);  unsqueeze_28 = None
        view_38: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_29, [1, 8192, 1024]);  unsqueeze_29 = None
        squeeze_dim_228: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_38, 0);  view_38 = None
        unsqueeze_30: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg17_1, 3);  arg17_1 = None
        unsqueeze_31: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 4);  unsqueeze_30 = None
        view_39: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_31, [1, 1024, 1024]);  unsqueeze_31 = None
        squeeze_dim_229: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_39, 0);  view_39 = None
        mm_default_114: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_228, squeeze_dim_229);  squeeze_dim_228 = squeeze_dim_229 = None
        unsqueeze_default_114: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_114, 0);  mm_default_114 = None
        view_40: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_114, [512, 16, 1, 16, 64]);  unsqueeze_default_114 = None
        permute_49: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_40, [0, 1, 3, 4, 2]);  view_40 = None
        view_41: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_49, [512, 16, 16, 64]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_11: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_41, arg21_1);  arg21_1 = None
        unsqueeze_44: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_11, 4);  add_11 = None
        permute_65: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_44, [1, 2, 0, 4, 3]);  unsqueeze_44 = None
        permute_67: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [0, 1, 2, 4, 3]);  permute_65 = None
        view_54: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_67, [256, 512, 64]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_32: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_32, 3)
        unsqueeze_33: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 4);  unsqueeze_32 = None
        view_42: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_33, [1, 8192, 1024]);  unsqueeze_33 = None
        squeeze_dim_226: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_42, 0);  view_42 = None
        unsqueeze_34: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg18_1, 3);  arg18_1 = None
        unsqueeze_35: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 4);  unsqueeze_34 = None
        view_43: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_35, [1, 1024, 1024]);  unsqueeze_35 = None
        squeeze_dim_227: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_43, 0);  view_43 = None
        mm_default_113: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_226, squeeze_dim_227);  squeeze_dim_226 = squeeze_dim_227 = None
        unsqueeze_default_113: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_113, 0);  mm_default_113 = None
        view_44: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_113, [512, 16, 1, 16, 64]);  unsqueeze_default_113 = None
        permute_54: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_44, [0, 1, 3, 4, 2]);  view_44 = None
        view_45: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_54, [512, 16, 16, 64]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_45: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_45, 4);  view_45 = None
        permute_66: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_45, [1, 2, 4, 0, 3]);  unsqueeze_45 = None
        permute_68: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [0, 1, 4, 3, 2]);  permute_66 = None
        view_55: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_68, [256, 64, 512]);  permute_68 = None
        bmm_12: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_54, view_55);  view_54 = view_55 = None
        view_56: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [16, 16, 512, 1, 512]);  bmm_12 = None
        permute_69: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 1, 2, 4, 3]);  view_56 = None
        view_57: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [16, 16, 512, 512]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_12: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_41, arg22_1);  view_41 = arg22_1 = None
        unsqueeze_46: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_12, 4);  add_12 = None
        permute_70: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_46, [1, 2, 0, 4, 3]);  unsqueeze_46 = None
        permute_72: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_70, [0, 1, 2, 4, 3]);  permute_70 = None
        view_58: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_72, [256, 512, 64]);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_39: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_40: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_39, 3);  convert_element_type_39 = None
        unsqueeze_41: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 4);  unsqueeze_40 = None
        view_50: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_41, [1, 16384, 1024]);  unsqueeze_41 = None
        squeeze_dim_222: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_50, 0);  view_50 = None
        unsqueeze_42: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg20_1, 3);  arg20_1 = None
        unsqueeze_43: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 4);  unsqueeze_42 = None
        view_51: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_43, [1, 1024, 1024]);  unsqueeze_43 = None
        squeeze_dim_223: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_51, 0);  view_51 = None
        mm_default_111: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_222, squeeze_dim_223);  squeeze_dim_222 = squeeze_dim_223 = None
        unsqueeze_default_111: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_111, 0);  mm_default_111 = None
        view_52: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_111, [1024, 16, 1, 16, 64]);  unsqueeze_default_111 = None
        permute_64: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 1, 3, 4, 2]);  view_52 = None
        view_53: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_64, [1024, 16, 16, 64]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_47: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_53, 4);  view_53 = None
        permute_71: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_47, [1, 2, 4, 0, 3]);  unsqueeze_47 = None
        permute_73: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [0, 1, 4, 3, 2]);  permute_71 = None
        view_59: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_73, [256, 64, 1024]);  permute_73 = None
        bmm_13: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [16, 16, 512, 1, 1024]);  bmm_13 = None
        permute_74: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_60, [0, 1, 2, 4, 3]);  view_60 = None
        view_61: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_74, [16, 16, 512, 1024]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_62: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_61, [16, 16, 1024, 512]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_4: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_62, 2, 1, 9223372036854775807);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_63: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_4, [16, 16, 512, 1023]);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_3: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_1: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_63, [None, None, None, iota_3]);  view_63 = iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_13: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_57, index_1);  view_57 = index_1 = None
        add_14: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, 0);  add_13 = None

        # No stacktrace found for following nodes
        mul_tensor_88: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, 0.125)
        convert_element_type_default_44: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_88, torch.float32);  mul_tensor_88 = None
        eq_tensor_22: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_44, convert_element_type_default_44)
        abs_default_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_44)
        ne_scalar_22: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_22, inf);  abs_default_22 = None
        mul_tensor_91: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_22, ne_scalar_22);  eq_tensor_22 = ne_scalar_22 = None
        logical_not_default_44: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_91);  mul_tensor_91 = None
        any_dims_22: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_44, [3], True);  logical_not_default_44 = None
        logical_not_default_45: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_22);  any_dims_22 = None
        convert_element_type_default_45: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        mul_tensor_89: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_45, 1);  convert_element_type_default_45 = None
        amax_default_44: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_89, [3], True)
        sub_tensor_44: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_89, amax_default_44);  mul_tensor_89 = amax_default_44 = None
        mul_tensor_90: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_44, 0.125);  sub_tensor_44 = None
        amax_default_45: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_44, [3], True)
        sub_tensor_45: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_44, amax_default_45);  convert_element_type_default_44 = amax_default_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_45, mul_tensor_90, sub_tensor_45);  logical_not_default_45 = mul_tensor_90 = sub_tensor_45 = None
        exp_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_22);  where_self_22 = None
        sum_2: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [3], True)
        div_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_47: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_48: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_47, 4);  convert_element_type_47 = None
        view_64: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_48, [256, 512, 512]);  unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_36: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_32, 3)
        unsqueeze_37: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 4);  unsqueeze_36 = None
        view_46: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_37, [1, 8192, 1024]);  unsqueeze_37 = None
        squeeze_dim_224: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_46, 0);  view_46 = None
        unsqueeze_38: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg19_1, 3);  arg19_1 = None
        unsqueeze_39: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 4);  unsqueeze_38 = None
        view_47: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_39, [1, 1024, 1024]);  unsqueeze_39 = None
        squeeze_dim_225: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_47, 0);  view_47 = None
        mm_default_112: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_224, squeeze_dim_225);  squeeze_dim_224 = squeeze_dim_225 = None
        unsqueeze_default_112: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_112, 0);  mm_default_112 = None
        view_48: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_112, [512, 16, 1, 16, 64]);  unsqueeze_default_112 = None
        permute_59: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_48, [0, 1, 3, 4, 2]);  view_48 = None
        view_49: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_59, [512, 16, 16, 64]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_49: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_49, 4);  view_49 = None
        permute_76: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_49, [4, 1, 2, 3, 0]);  unsqueeze_49 = None
        permute_78: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 2, 4, 3, 0]);  permute_76 = None
        view_65: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_78, [256, 512, 64]);  permute_78 = None
        bmm_14: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_64, view_65);  view_64 = view_65 = None
        view_66: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [16, 16, 512, 1, 64]);  bmm_14 = None
        permute_79: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_66, [2, 0, 1, 4, 3]);  view_66 = None
        view_67: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_79, [512, 16, 16, 64]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_50: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_67, 4);  view_67 = None
        permute_80: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_50, [0, 1, 4, 3, 2]);  unsqueeze_50 = None
        permute_82: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 4, 2]);  permute_80 = None
        clone_10: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_68: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 8192, 1024]);  clone_10 = None
        squeeze_dim_220: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_68, 0);  view_68 = None
        unsqueeze_51: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg23_1, 3);  arg23_1 = None
        unsqueeze_52: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 4);  unsqueeze_51 = None
        permute_81: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_52, [3, 4, 0, 2, 1]);  unsqueeze_52 = None
        permute_83: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_81, [3, 4, 2, 0, 1]);  permute_81 = None
        clone_11: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_69: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [1, 1024, 1024]);  clone_11 = None
        squeeze_dim_221: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_69, 0);  view_69 = None
        mm_default_110: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_220, squeeze_dim_221);  squeeze_dim_220 = squeeze_dim_221 = None
        unsqueeze_default_110: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_110, 0);  mm_default_110 = None
        view_70: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_110, [512, 16, 1, 1, 1024]);  unsqueeze_default_110 = None
        permute_84: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_70, [0, 1, 4, 2, 3]);  view_70 = None
        view_71: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_84, [512, 16, 1024]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_15: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_71, convert_element_type_32);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_52: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_52, [2], correction = 0, keepdim = True)
        getitem_4: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_4: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_52, getitem_5);  convert_element_type_52 = getitem_5 = None
        add_16: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_11: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_12: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, arg24_1);  mul_11 = arg24_1 = None
        add_17: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_12, arg25_1);  mul_12 = arg25_1 = None
        convert_element_type_53: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_72: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_53, [8192, 1024])
        permute_85: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_2: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg27_1, view_72, permute_85);  arg27_1 = view_72 = permute_85 = None
        view_73: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [512, 16, 4096]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_57: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        mul_13: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_57, 0.5)
        mul_14: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_57, 0.7071067811865476);  convert_element_type_57 = None
        erf_1: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_18: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_15: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, add_18);  mul_13 = add_18 = None
        convert_element_type_58: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_74: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_58, [8192, 4096]);  convert_element_type_58 = None
        permute_86: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_3: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg29_1, view_74, permute_86);  arg29_1 = view_74 = permute_86 = None
        view_75: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [512, 16, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_19: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_75, convert_element_type_53);  view_75 = convert_element_type_53 = None
        convert_element_type_62: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.float32);  add_19 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_62, [2], correction = 0, keepdim = True)
        getitem_6: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_7);  convert_element_type_62 = getitem_7 = None
        add_20: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_16: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_17: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, arg30_1);  mul_16 = arg30_1 = None
        add_21: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, arg31_1);  mul_17 = arg31_1 = None
        convert_element_type_63: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_53: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_63, 3)
        unsqueeze_54: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 4);  unsqueeze_53 = None
        view_76: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_54, [1, 8192, 1024]);  unsqueeze_54 = None
        squeeze_dim_218: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_76, 0);  view_76 = None
        unsqueeze_55: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg32_1, 3);  arg32_1 = None
        unsqueeze_56: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
        view_77: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_56, [1, 1024, 1024]);  unsqueeze_56 = None
        squeeze_dim_219: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_77, 0);  view_77 = None
        mm_default_109: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_218, squeeze_dim_219);  squeeze_dim_218 = squeeze_dim_219 = None
        unsqueeze_default_109: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_109, 0);  mm_default_109 = None
        view_78: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_109, [512, 16, 1, 16, 64]);  unsqueeze_default_109 = None
        permute_91: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 1, 3, 4, 2]);  view_78 = None
        view_79: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_91, [512, 16, 16, 64]);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_22: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_79, arg36_1);  arg36_1 = None
        unsqueeze_69: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_22, 4);  add_22 = None
        permute_107: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_69, [1, 2, 0, 4, 3]);  unsqueeze_69 = None
        permute_109: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [0, 1, 2, 4, 3]);  permute_107 = None
        view_92: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_109, [256, 512, 64]);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_57: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_63, 3)
        unsqueeze_58: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 4);  unsqueeze_57 = None
        view_80: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_58, [1, 8192, 1024]);  unsqueeze_58 = None
        squeeze_dim_216: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_80, 0);  view_80 = None
        unsqueeze_59: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg33_1, 3);  arg33_1 = None
        unsqueeze_60: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 4);  unsqueeze_59 = None
        view_81: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_60, [1, 1024, 1024]);  unsqueeze_60 = None
        squeeze_dim_217: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_81, 0);  view_81 = None
        mm_default_108: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_216, squeeze_dim_217);  squeeze_dim_216 = squeeze_dim_217 = None
        unsqueeze_default_108: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_108, 0);  mm_default_108 = None
        view_82: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_108, [512, 16, 1, 16, 64]);  unsqueeze_default_108 = None
        permute_96: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_82, [0, 1, 3, 4, 2]);  view_82 = None
        view_83: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_96, [512, 16, 16, 64]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_70: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_83, 4);  view_83 = None
        permute_108: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_70, [1, 2, 4, 0, 3]);  unsqueeze_70 = None
        permute_110: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_108, [0, 1, 4, 3, 2]);  permute_108 = None
        view_93: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_110, [256, 64, 512]);  permute_110 = None
        bmm_20: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_92, view_93);  view_92 = view_93 = None
        view_94: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [16, 16, 512, 1, 512]);  bmm_20 = None
        permute_111: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_94, [0, 1, 2, 4, 3]);  view_94 = None
        view_95: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_111, [16, 16, 512, 512]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_23: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_79, arg37_1);  view_79 = arg37_1 = None
        unsqueeze_71: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_23, 4);  add_23 = None
        permute_112: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_71, [1, 2, 0, 4, 3]);  unsqueeze_71 = None
        permute_114: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [0, 1, 2, 4, 3]);  permute_112 = None
        view_96: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_114, [256, 512, 64]);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_70: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_65: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_70, 3);  convert_element_type_70 = None
        unsqueeze_66: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 4);  unsqueeze_65 = None
        view_88: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_66, [1, 16384, 1024]);  unsqueeze_66 = None
        squeeze_dim_212: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_88, 0);  view_88 = None
        unsqueeze_67: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg35_1, 3);  arg35_1 = None
        unsqueeze_68: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 4);  unsqueeze_67 = None
        view_89: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_68, [1, 1024, 1024]);  unsqueeze_68 = None
        squeeze_dim_213: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_89, 0);  view_89 = None
        mm_default_106: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_212, squeeze_dim_213);  squeeze_dim_212 = squeeze_dim_213 = None
        unsqueeze_default_106: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_106, 0);  mm_default_106 = None
        view_90: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_106, [1024, 16, 1, 16, 64]);  unsqueeze_default_106 = None
        permute_106: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 1, 3, 4, 2]);  view_90 = None
        view_91: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_106, [1024, 16, 16, 64]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_72: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_91, 4);  view_91 = None
        permute_113: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_72, [1, 2, 4, 0, 3]);  unsqueeze_72 = None
        permute_115: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_113, [0, 1, 4, 3, 2]);  permute_113 = None
        view_97: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_115, [256, 64, 1024]);  permute_115 = None
        bmm_21: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_96, view_97);  view_96 = view_97 = None
        view_98: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [16, 16, 512, 1, 1024]);  bmm_21 = None
        permute_116: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 1, 2, 4, 3]);  view_98 = None
        view_99: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_116, [16, 16, 512, 1024]);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_100: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_99, [16, 16, 1024, 512]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_6: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_100, 2, 1, 9223372036854775807);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_101: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_6, [16, 16, 512, 1023]);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_4: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_2: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_101, [None, None, None, iota_4]);  view_101 = iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_24: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_95, index_2);  view_95 = index_2 = None
        add_25: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_24, 0);  add_24 = None

        # No stacktrace found for following nodes
        mul_tensor_84: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_25, 0.125)
        convert_element_type_default_42: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_84, torch.float32);  mul_tensor_84 = None
        eq_tensor_21: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_42, convert_element_type_default_42)
        abs_default_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_42)
        ne_scalar_21: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_21, inf);  abs_default_21 = None
        mul_tensor_87: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_21, ne_scalar_21);  eq_tensor_21 = ne_scalar_21 = None
        logical_not_default_42: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_87);  mul_tensor_87 = None
        any_dims_21: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_42, [3], True);  logical_not_default_42 = None
        logical_not_default_43: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_21);  any_dims_21 = None
        convert_element_type_default_43: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        mul_tensor_85: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_43, 1);  convert_element_type_default_43 = None
        amax_default_42: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_85, [3], True)
        sub_tensor_42: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_85, amax_default_42);  mul_tensor_85 = amax_default_42 = None
        mul_tensor_86: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_42, 0.125);  sub_tensor_42 = None
        amax_default_43: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_42, [3], True)
        sub_tensor_43: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_42, amax_default_43);  convert_element_type_default_42 = amax_default_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_43, mul_tensor_86, sub_tensor_43);  logical_not_default_43 = mul_tensor_86 = sub_tensor_43 = None
        exp_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_21);  where_self_21 = None
        sum_3: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [3], True)
        div_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_78: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_73: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_78, 4);  convert_element_type_78 = None
        view_102: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_73, [256, 512, 512]);  unsqueeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_61: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_63, 3)
        unsqueeze_62: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 4);  unsqueeze_61 = None
        view_84: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_62, [1, 8192, 1024]);  unsqueeze_62 = None
        squeeze_dim_214: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_84, 0);  view_84 = None
        unsqueeze_63: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg34_1, 3);  arg34_1 = None
        unsqueeze_64: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 4);  unsqueeze_63 = None
        view_85: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_64, [1, 1024, 1024]);  unsqueeze_64 = None
        squeeze_dim_215: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_85, 0);  view_85 = None
        mm_default_107: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_214, squeeze_dim_215);  squeeze_dim_214 = squeeze_dim_215 = None
        unsqueeze_default_107: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_107, 0);  mm_default_107 = None
        view_86: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_107, [512, 16, 1, 16, 64]);  unsqueeze_default_107 = None
        permute_101: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_86, [0, 1, 3, 4, 2]);  view_86 = None
        view_87: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_101, [512, 16, 16, 64]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_74: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_87, 4);  view_87 = None
        permute_118: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_74, [4, 1, 2, 3, 0]);  unsqueeze_74 = None
        permute_120: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 2, 4, 3, 0]);  permute_118 = None
        view_103: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_120, [256, 512, 64]);  permute_120 = None
        bmm_22: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
        view_104: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [16, 16, 512, 1, 64]);  bmm_22 = None
        permute_121: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_104, [2, 0, 1, 4, 3]);  view_104 = None
        view_105: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_121, [512, 16, 16, 64]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_75: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_105, 4);  view_105 = None
        permute_122: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 3, 2]);  unsqueeze_75 = None
        permute_124: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 4, 2]);  permute_122 = None
        clone_16: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_106: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [1, 8192, 1024]);  clone_16 = None
        squeeze_dim_210: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_106, 0);  view_106 = None
        unsqueeze_76: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg38_1, 3);  arg38_1 = None
        unsqueeze_77: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 4);  unsqueeze_76 = None
        permute_123: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_77, [3, 4, 0, 2, 1]);  unsqueeze_77 = None
        permute_125: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [3, 4, 2, 0, 1]);  permute_123 = None
        clone_17: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_107: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1, 1024, 1024]);  clone_17 = None
        squeeze_dim_211: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_107, 0);  view_107 = None
        mm_default_105: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_210, squeeze_dim_211);  squeeze_dim_210 = squeeze_dim_211 = None
        unsqueeze_default_105: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_105, 0);  mm_default_105 = None
        view_108: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_105, [512, 16, 1, 1, 1024]);  unsqueeze_default_105 = None
        permute_126: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_108, [0, 1, 4, 2, 3]);  view_108 = None
        view_109: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_126, [512, 16, 1024]);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_26: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_109, convert_element_type_63);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_83: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.float32);  add_26 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_83, [2], correction = 0, keepdim = True)
        getitem_8: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_7: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_83, getitem_9);  convert_element_type_83 = getitem_9 = None
        add_27: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_19: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_20: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, arg39_1);  mul_19 = arg39_1 = None
        add_28: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, arg40_1);  mul_20 = arg40_1 = None
        convert_element_type_84: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_110: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_84, [8192, 1024])
        permute_127: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_4: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_110, permute_127);  arg42_1 = view_110 = permute_127 = None
        view_111: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [512, 16, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_88: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_111, torch.float32);  view_111 = None
        mul_21: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.5)
        mul_22: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.7071067811865476);  convert_element_type_88 = None
        erf_2: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
        add_29: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_23: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, add_29);  mul_21 = add_29 = None
        convert_element_type_89: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_23, torch.bfloat16);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_112: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_89, [8192, 4096]);  convert_element_type_89 = None
        permute_128: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_5: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_112, permute_128);  arg44_1 = view_112 = permute_128 = None
        view_113: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [512, 16, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_30: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_113, convert_element_type_84);  view_113 = convert_element_type_84 = None
        convert_element_type_93: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_93, [2], correction = 0, keepdim = True)
        getitem_10: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_93, getitem_11);  convert_element_type_93 = getitem_11 = None
        add_31: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_24: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_25: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, arg45_1);  mul_24 = arg45_1 = None
        add_32: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, arg46_1);  mul_25 = arg46_1 = None
        convert_element_type_94: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_78: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_94, 3)
        unsqueeze_79: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 4);  unsqueeze_78 = None
        view_114: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_79, [1, 8192, 1024]);  unsqueeze_79 = None
        squeeze_dim_208: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_114, 0);  view_114 = None
        unsqueeze_80: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg47_1, 3);  arg47_1 = None
        unsqueeze_81: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 4);  unsqueeze_80 = None
        view_115: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_81, [1, 1024, 1024]);  unsqueeze_81 = None
        squeeze_dim_209: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_115, 0);  view_115 = None
        mm_default_104: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_208, squeeze_dim_209);  squeeze_dim_208 = squeeze_dim_209 = None
        unsqueeze_default_104: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_104, 0);  mm_default_104 = None
        view_116: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_104, [512, 16, 1, 16, 64]);  unsqueeze_default_104 = None
        permute_133: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 1, 3, 4, 2]);  view_116 = None
        view_117: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_133, [512, 16, 16, 64]);  permute_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_33: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_117, arg51_1);  arg51_1 = None
        unsqueeze_94: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_33, 4);  add_33 = None
        permute_149: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_94, [1, 2, 0, 4, 3]);  unsqueeze_94 = None
        permute_151: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_149, [0, 1, 2, 4, 3]);  permute_149 = None
        view_130: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_151, [256, 512, 64]);  permute_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_82: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_94, 3)
        unsqueeze_83: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 4);  unsqueeze_82 = None
        view_118: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_83, [1, 8192, 1024]);  unsqueeze_83 = None
        squeeze_dim_206: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_118, 0);  view_118 = None
        unsqueeze_84: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg48_1, 3);  arg48_1 = None
        unsqueeze_85: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 4);  unsqueeze_84 = None
        view_119: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_85, [1, 1024, 1024]);  unsqueeze_85 = None
        squeeze_dim_207: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_119, 0);  view_119 = None
        mm_default_103: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_206, squeeze_dim_207);  squeeze_dim_206 = squeeze_dim_207 = None
        unsqueeze_default_103: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_103, 0);  mm_default_103 = None
        view_120: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_103, [512, 16, 1, 16, 64]);  unsqueeze_default_103 = None
        permute_138: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 1, 3, 4, 2]);  view_120 = None
        view_121: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_138, [512, 16, 16, 64]);  permute_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_95: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_121, 4);  view_121 = None
        permute_150: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_95, [1, 2, 4, 0, 3]);  unsqueeze_95 = None
        permute_152: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_150, [0, 1, 4, 3, 2]);  permute_150 = None
        view_131: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_152, [256, 64, 512]);  permute_152 = None
        bmm_28: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
        view_132: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [16, 16, 512, 1, 512]);  bmm_28 = None
        permute_153: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 1, 2, 4, 3]);  view_132 = None
        view_133: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_153, [16, 16, 512, 512]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_34: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_117, arg52_1);  view_117 = arg52_1 = None
        unsqueeze_96: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_34, 4);  add_34 = None
        permute_154: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_96, [1, 2, 0, 4, 3]);  unsqueeze_96 = None
        permute_156: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_154, [0, 1, 2, 4, 3]);  permute_154 = None
        view_134: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_156, [256, 512, 64]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_101: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_90: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_101, 3);  convert_element_type_101 = None
        unsqueeze_91: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 4);  unsqueeze_90 = None
        view_126: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_91, [1, 16384, 1024]);  unsqueeze_91 = None
        squeeze_dim_202: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_126, 0);  view_126 = None
        unsqueeze_92: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg50_1, 3);  arg50_1 = None
        unsqueeze_93: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 4);  unsqueeze_92 = None
        view_127: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_93, [1, 1024, 1024]);  unsqueeze_93 = None
        squeeze_dim_203: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_127, 0);  view_127 = None
        mm_default_101: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_202, squeeze_dim_203);  squeeze_dim_202 = squeeze_dim_203 = None
        unsqueeze_default_101: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_101, 0);  mm_default_101 = None
        view_128: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_101, [1024, 16, 1, 16, 64]);  unsqueeze_default_101 = None
        permute_148: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_128, [0, 1, 3, 4, 2]);  view_128 = None
        view_129: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_148, [1024, 16, 16, 64]);  permute_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_97: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_129, 4);  view_129 = None
        permute_155: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_97, [1, 2, 4, 0, 3]);  unsqueeze_97 = None
        permute_157: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_155, [0, 1, 4, 3, 2]);  permute_155 = None
        view_135: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_157, [256, 64, 1024]);  permute_157 = None
        bmm_29: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_134, view_135);  view_134 = view_135 = None
        view_136: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [16, 16, 512, 1, 1024]);  bmm_29 = None
        permute_158: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_136, [0, 1, 2, 4, 3]);  view_136 = None
        view_137: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_158, [16, 16, 512, 1024]);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_138: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [16, 16, 1024, 512]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_8: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_138, 2, 1, 9223372036854775807);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_139: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_8, [16, 16, 512, 1023]);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_5: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_3: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_139, [None, None, None, iota_5]);  view_139 = iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_35: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_133, index_3);  view_133 = index_3 = None
        add_36: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_35, 0);  add_35 = None

        # No stacktrace found for following nodes
        mul_tensor_80: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_36, 0.125)
        convert_element_type_default_40: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_80, torch.float32);  mul_tensor_80 = None
        eq_tensor_20: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_40, convert_element_type_default_40)
        abs_default_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_40)
        ne_scalar_20: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_20, inf);  abs_default_20 = None
        mul_tensor_83: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_20, ne_scalar_20);  eq_tensor_20 = ne_scalar_20 = None
        logical_not_default_40: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_83);  mul_tensor_83 = None
        any_dims_20: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_40, [3], True);  logical_not_default_40 = None
        logical_not_default_41: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_20);  any_dims_20 = None
        convert_element_type_default_41: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.float32);  add_36 = None
        mul_tensor_81: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_41, 1);  convert_element_type_default_41 = None
        amax_default_40: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_81, [3], True)
        sub_tensor_40: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_81, amax_default_40);  mul_tensor_81 = amax_default_40 = None
        mul_tensor_82: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_40, 0.125);  sub_tensor_40 = None
        amax_default_41: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_40, [3], True)
        sub_tensor_41: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_40, amax_default_41);  convert_element_type_default_40 = amax_default_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_41, mul_tensor_82, sub_tensor_41);  logical_not_default_41 = mul_tensor_82 = sub_tensor_41 = None
        exp_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_20);  where_self_20 = None
        sum_4: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [3], True)
        div_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_109: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_98: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_109, 4);  convert_element_type_109 = None
        view_140: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_98, [256, 512, 512]);  unsqueeze_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_86: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_94, 3)
        unsqueeze_87: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, 4);  unsqueeze_86 = None
        view_122: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_87, [1, 8192, 1024]);  unsqueeze_87 = None
        squeeze_dim_204: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_122, 0);  view_122 = None
        unsqueeze_88: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg49_1, 3);  arg49_1 = None
        unsqueeze_89: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 4);  unsqueeze_88 = None
        view_123: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_89, [1, 1024, 1024]);  unsqueeze_89 = None
        squeeze_dim_205: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_123, 0);  view_123 = None
        mm_default_102: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_204, squeeze_dim_205);  squeeze_dim_204 = squeeze_dim_205 = None
        unsqueeze_default_102: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_102, 0);  mm_default_102 = None
        view_124: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_102, [512, 16, 1, 16, 64]);  unsqueeze_default_102 = None
        permute_143: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 1, 3, 4, 2]);  view_124 = None
        view_125: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_143, [512, 16, 16, 64]);  permute_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_99: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_125, 4);  view_125 = None
        permute_160: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_99, [4, 1, 2, 3, 0]);  unsqueeze_99 = None
        permute_162: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_160, [1, 2, 4, 3, 0]);  permute_160 = None
        view_141: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_162, [256, 512, 64]);  permute_162 = None
        bmm_30: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
        view_142: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [16, 16, 512, 1, 64]);  bmm_30 = None
        permute_163: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_142, [2, 0, 1, 4, 3]);  view_142 = None
        view_143: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_163, [512, 16, 16, 64]);  permute_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_100: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_143, 4);  view_143 = None
        permute_164: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_100, [0, 1, 4, 3, 2]);  unsqueeze_100 = None
        permute_166: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 4, 2]);  permute_164 = None
        clone_22: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_144: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [1, 8192, 1024]);  clone_22 = None
        squeeze_dim_200: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_144, 0);  view_144 = None
        unsqueeze_101: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg53_1, 3);  arg53_1 = None
        unsqueeze_102: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 4);  unsqueeze_101 = None
        permute_165: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_102, [3, 4, 0, 2, 1]);  unsqueeze_102 = None
        permute_167: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_165, [3, 4, 2, 0, 1]);  permute_165 = None
        clone_23: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_145: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [1, 1024, 1024]);  clone_23 = None
        squeeze_dim_201: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_145, 0);  view_145 = None
        mm_default_100: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_200, squeeze_dim_201);  squeeze_dim_200 = squeeze_dim_201 = None
        unsqueeze_default_100: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_100, 0);  mm_default_100 = None
        view_146: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_100, [512, 16, 1, 1, 1024]);  unsqueeze_default_100 = None
        permute_168: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 1, 4, 2, 3]);  view_146 = None
        view_147: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_168, [512, 16, 1024]);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_37: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_147, convert_element_type_94);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_114: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_114, [2], correction = 0, keepdim = True)
        getitem_12: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_10: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_114, getitem_13);  convert_element_type_114 = getitem_13 = None
        add_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_27: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_28: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, arg54_1);  mul_27 = arg54_1 = None
        add_39: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_28, arg55_1);  mul_28 = arg55_1 = None
        convert_element_type_115: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_148: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_115, [8192, 1024])
        permute_169: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_6: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg57_1, view_148, permute_169);  arg57_1 = view_148 = permute_169 = None
        view_149: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [512, 16, 4096]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_119: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_149, torch.float32);  view_149 = None
        mul_29: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, 0.5)
        mul_30: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, 0.7071067811865476);  convert_element_type_119 = None
        erf_3: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
        add_40: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_31: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, add_40);  mul_29 = add_40 = None
        convert_element_type_120: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_150: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_120, [8192, 4096]);  convert_element_type_120 = None
        permute_170: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_7: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg59_1, view_150, permute_170);  arg59_1 = view_150 = permute_170 = None
        view_151: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [512, 16, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_41: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_151, convert_element_type_115);  view_151 = convert_element_type_115 = None
        convert_element_type_124: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_124, [2], correction = 0, keepdim = True)
        getitem_14: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_124, getitem_15);  convert_element_type_124 = getitem_15 = None
        add_42: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_32: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_33: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, arg60_1);  mul_32 = arg60_1 = None
        add_43: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, arg61_1);  mul_33 = arg61_1 = None
        convert_element_type_125: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_103: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_125, 3)
        unsqueeze_104: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 4);  unsqueeze_103 = None
        view_152: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_104, [1, 8192, 1024]);  unsqueeze_104 = None
        squeeze_dim_198: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_152, 0);  view_152 = None
        unsqueeze_105: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg62_1, 3);  arg62_1 = None
        unsqueeze_106: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 4);  unsqueeze_105 = None
        view_153: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_106, [1, 1024, 1024]);  unsqueeze_106 = None
        squeeze_dim_199: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_153, 0);  view_153 = None
        mm_default_99: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_198, squeeze_dim_199);  squeeze_dim_198 = squeeze_dim_199 = None
        unsqueeze_default_99: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_99, 0);  mm_default_99 = None
        view_154: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_99, [512, 16, 1, 16, 64]);  unsqueeze_default_99 = None
        permute_175: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_154, [0, 1, 3, 4, 2]);  view_154 = None
        view_155: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_175, [512, 16, 16, 64]);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_44: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_155, arg66_1);  arg66_1 = None
        unsqueeze_119: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_44, 4);  add_44 = None
        permute_191: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_119, [1, 2, 0, 4, 3]);  unsqueeze_119 = None
        permute_193: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_191, [0, 1, 2, 4, 3]);  permute_191 = None
        view_168: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_193, [256, 512, 64]);  permute_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_107: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_125, 3)
        unsqueeze_108: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 4);  unsqueeze_107 = None
        view_156: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_108, [1, 8192, 1024]);  unsqueeze_108 = None
        squeeze_dim_196: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_156, 0);  view_156 = None
        unsqueeze_109: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg63_1, 3);  arg63_1 = None
        unsqueeze_110: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 4);  unsqueeze_109 = None
        view_157: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_110, [1, 1024, 1024]);  unsqueeze_110 = None
        squeeze_dim_197: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_157, 0);  view_157 = None
        mm_default_98: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_196, squeeze_dim_197);  squeeze_dim_196 = squeeze_dim_197 = None
        unsqueeze_default_98: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_98, 0);  mm_default_98 = None
        view_158: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_98, [512, 16, 1, 16, 64]);  unsqueeze_default_98 = None
        permute_180: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_158, [0, 1, 3, 4, 2]);  view_158 = None
        view_159: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_180, [512, 16, 16, 64]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_120: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_159, 4);  view_159 = None
        permute_192: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_120, [1, 2, 4, 0, 3]);  unsqueeze_120 = None
        permute_194: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_192, [0, 1, 4, 3, 2]);  permute_192 = None
        view_169: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_194, [256, 64, 512]);  permute_194 = None
        bmm_36: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
        view_170: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [16, 16, 512, 1, 512]);  bmm_36 = None
        permute_195: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_170, [0, 1, 2, 4, 3]);  view_170 = None
        view_171: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_195, [16, 16, 512, 512]);  permute_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_45: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_155, arg67_1);  view_155 = arg67_1 = None
        unsqueeze_121: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_45, 4);  add_45 = None
        permute_196: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_121, [1, 2, 0, 4, 3]);  unsqueeze_121 = None
        permute_198: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_196, [0, 1, 2, 4, 3]);  permute_196 = None
        view_172: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_198, [256, 512, 64]);  permute_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_132: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_115: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_132, 3);  convert_element_type_132 = None
        unsqueeze_116: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 4);  unsqueeze_115 = None
        view_164: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_116, [1, 16384, 1024]);  unsqueeze_116 = None
        squeeze_dim_192: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_164, 0);  view_164 = None
        unsqueeze_117: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg65_1, 3);  arg65_1 = None
        unsqueeze_118: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 4);  unsqueeze_117 = None
        view_165: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_118, [1, 1024, 1024]);  unsqueeze_118 = None
        squeeze_dim_193: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_165, 0);  view_165 = None
        mm_default_96: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_192, squeeze_dim_193);  squeeze_dim_192 = squeeze_dim_193 = None
        unsqueeze_default_96: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_96, 0);  mm_default_96 = None
        view_166: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_96, [1024, 16, 1, 16, 64]);  unsqueeze_default_96 = None
        permute_190: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 1, 3, 4, 2]);  view_166 = None
        view_167: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_190, [1024, 16, 16, 64]);  permute_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_122: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_167, 4);  view_167 = None
        permute_197: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_122, [1, 2, 4, 0, 3]);  unsqueeze_122 = None
        permute_199: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_197, [0, 1, 4, 3, 2]);  permute_197 = None
        view_173: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_199, [256, 64, 1024]);  permute_199 = None
        bmm_37: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_172, view_173);  view_172 = view_173 = None
        view_174: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [16, 16, 512, 1, 1024]);  bmm_37 = None
        permute_200: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_174, [0, 1, 2, 4, 3]);  view_174 = None
        view_175: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_200, [16, 16, 512, 1024]);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_176: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_175, [16, 16, 1024, 512]);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_10: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_176, 2, 1, 9223372036854775807);  view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_177: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_10, [16, 16, 512, 1023]);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_6: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_4: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_177, [None, None, None, iota_6]);  view_177 = iota_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_46: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_171, index_4);  view_171 = index_4 = None
        add_47: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_46, 0);  add_46 = None

        # No stacktrace found for following nodes
        mul_tensor_76: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_47, 0.125)
        convert_element_type_default_38: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_76, torch.float32);  mul_tensor_76 = None
        eq_tensor_19: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_38, convert_element_type_default_38)
        abs_default_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_38)
        ne_scalar_19: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_19, inf);  abs_default_19 = None
        mul_tensor_79: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_19, ne_scalar_19);  eq_tensor_19 = ne_scalar_19 = None
        logical_not_default_38: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_79);  mul_tensor_79 = None
        any_dims_19: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_38, [3], True);  logical_not_default_38 = None
        logical_not_default_39: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_19);  any_dims_19 = None
        convert_element_type_default_39: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        mul_tensor_77: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_39, 1);  convert_element_type_default_39 = None
        amax_default_38: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_77, [3], True)
        sub_tensor_38: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_77, amax_default_38);  mul_tensor_77 = amax_default_38 = None
        mul_tensor_78: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_38, 0.125);  sub_tensor_38 = None
        amax_default_39: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_38, [3], True)
        sub_tensor_39: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_38, amax_default_39);  convert_element_type_default_38 = amax_default_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_39, mul_tensor_78, sub_tensor_39);  logical_not_default_39 = mul_tensor_78 = sub_tensor_39 = None
        exp_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_19);  where_self_19 = None
        sum_5: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [3], True)
        div_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_140: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_123: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_140, 4);  convert_element_type_140 = None
        view_178: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_123, [256, 512, 512]);  unsqueeze_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_111: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_125, 3)
        unsqueeze_112: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 4);  unsqueeze_111 = None
        view_160: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_112, [1, 8192, 1024]);  unsqueeze_112 = None
        squeeze_dim_194: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_160, 0);  view_160 = None
        unsqueeze_113: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg64_1, 3);  arg64_1 = None
        unsqueeze_114: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 4);  unsqueeze_113 = None
        view_161: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_114, [1, 1024, 1024]);  unsqueeze_114 = None
        squeeze_dim_195: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_161, 0);  view_161 = None
        mm_default_97: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_194, squeeze_dim_195);  squeeze_dim_194 = squeeze_dim_195 = None
        unsqueeze_default_97: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_97, 0);  mm_default_97 = None
        view_162: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_97, [512, 16, 1, 16, 64]);  unsqueeze_default_97 = None
        permute_185: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 1, 3, 4, 2]);  view_162 = None
        view_163: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_185, [512, 16, 16, 64]);  permute_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_124: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_163, 4);  view_163 = None
        permute_202: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_124, [4, 1, 2, 3, 0]);  unsqueeze_124 = None
        permute_204: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_202, [1, 2, 4, 3, 0]);  permute_202 = None
        view_179: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_204, [256, 512, 64]);  permute_204 = None
        bmm_38: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_178, view_179);  view_178 = view_179 = None
        view_180: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [16, 16, 512, 1, 64]);  bmm_38 = None
        permute_205: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_180, [2, 0, 1, 4, 3]);  view_180 = None
        view_181: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_205, [512, 16, 16, 64]);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_125: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_181, 4);  view_181 = None
        permute_206: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_125, [0, 1, 4, 3, 2]);  unsqueeze_125 = None
        permute_208: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 4, 2]);  permute_206 = None
        clone_28: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
        view_182: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [1, 8192, 1024]);  clone_28 = None
        squeeze_dim_190: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_182, 0);  view_182 = None
        unsqueeze_126: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg68_1, 3);  arg68_1 = None
        unsqueeze_127: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 4);  unsqueeze_126 = None
        permute_207: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_127, [3, 4, 0, 2, 1]);  unsqueeze_127 = None
        permute_209: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_207, [3, 4, 2, 0, 1]);  permute_207 = None
        clone_29: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        view_183: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [1, 1024, 1024]);  clone_29 = None
        squeeze_dim_191: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_183, 0);  view_183 = None
        mm_default_95: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_190, squeeze_dim_191);  squeeze_dim_190 = squeeze_dim_191 = None
        unsqueeze_default_95: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_95, 0);  mm_default_95 = None
        view_184: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_95, [512, 16, 1, 1, 1024]);  unsqueeze_default_95 = None
        permute_210: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 1, 4, 2, 3]);  view_184 = None
        view_185: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_210, [512, 16, 1024]);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_48: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_185, convert_element_type_125);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_145: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_145, [2], correction = 0, keepdim = True)
        getitem_16: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_13: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_145, getitem_17);  convert_element_type_145 = getitem_17 = None
        add_49: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_35: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_36: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, arg69_1);  mul_35 = arg69_1 = None
        add_50: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, arg70_1);  mul_36 = arg70_1 = None
        convert_element_type_146: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_186: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_146, [8192, 1024])
        permute_211: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_8: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_186, permute_211);  arg72_1 = view_186 = permute_211 = None
        view_187: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [512, 16, 4096]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_150: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_187, torch.float32);  view_187 = None
        mul_37: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_150, 0.5)
        mul_38: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_150, 0.7071067811865476);  convert_element_type_150 = None
        erf_4: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_51: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_39: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, add_51);  mul_37 = add_51 = None
        convert_element_type_151: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_188: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [8192, 4096]);  convert_element_type_151 = None
        permute_212: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_9: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_188, permute_212);  arg74_1 = view_188 = permute_212 = None
        view_189: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [512, 16, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_52: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_189, convert_element_type_146);  view_189 = convert_element_type_146 = None
        convert_element_type_155: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.float32);  add_52 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_155, [2], correction = 0, keepdim = True)
        getitem_18: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_155, getitem_19);  convert_element_type_155 = getitem_19 = None
        add_53: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_40: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_41: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, arg75_1);  mul_40 = arg75_1 = None
        add_54: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, arg76_1);  mul_41 = arg76_1 = None
        convert_element_type_156: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_128: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_156, 3)
        unsqueeze_129: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 4);  unsqueeze_128 = None
        view_190: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_129, [1, 8192, 1024]);  unsqueeze_129 = None
        squeeze_dim_188: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_190, 0);  view_190 = None
        unsqueeze_130: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg77_1, 3);  arg77_1 = None
        unsqueeze_131: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 4);  unsqueeze_130 = None
        view_191: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_131, [1, 1024, 1024]);  unsqueeze_131 = None
        squeeze_dim_189: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_191, 0);  view_191 = None
        mm_default_94: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_188, squeeze_dim_189);  squeeze_dim_188 = squeeze_dim_189 = None
        unsqueeze_default_94: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_94, 0);  mm_default_94 = None
        view_192: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_94, [512, 16, 1, 16, 64]);  unsqueeze_default_94 = None
        permute_217: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 1, 3, 4, 2]);  view_192 = None
        view_193: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_217, [512, 16, 16, 64]);  permute_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_55: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, arg81_1);  arg81_1 = None
        unsqueeze_144: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_55, 4);  add_55 = None
        permute_233: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_144, [1, 2, 0, 4, 3]);  unsqueeze_144 = None
        permute_235: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_233, [0, 1, 2, 4, 3]);  permute_233 = None
        view_206: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_235, [256, 512, 64]);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_132: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_156, 3)
        unsqueeze_133: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 4);  unsqueeze_132 = None
        view_194: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_133, [1, 8192, 1024]);  unsqueeze_133 = None
        squeeze_dim_186: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_194, 0);  view_194 = None
        unsqueeze_134: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg78_1, 3);  arg78_1 = None
        unsqueeze_135: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 4);  unsqueeze_134 = None
        view_195: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_135, [1, 1024, 1024]);  unsqueeze_135 = None
        squeeze_dim_187: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_195, 0);  view_195 = None
        mm_default_93: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_186, squeeze_dim_187);  squeeze_dim_186 = squeeze_dim_187 = None
        unsqueeze_default_93: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_93, 0);  mm_default_93 = None
        view_196: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_93, [512, 16, 1, 16, 64]);  unsqueeze_default_93 = None
        permute_222: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_196, [0, 1, 3, 4, 2]);  view_196 = None
        view_197: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_222, [512, 16, 16, 64]);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_145: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_197, 4);  view_197 = None
        permute_234: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_145, [1, 2, 4, 0, 3]);  unsqueeze_145 = None
        permute_236: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_234, [0, 1, 4, 3, 2]);  permute_234 = None
        view_207: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_236, [256, 64, 512]);  permute_236 = None
        bmm_44: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_206, view_207);  view_206 = view_207 = None
        view_208: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [16, 16, 512, 1, 512]);  bmm_44 = None
        permute_237: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_208, [0, 1, 2, 4, 3]);  view_208 = None
        view_209: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_237, [16, 16, 512, 512]);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_56: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, arg82_1);  view_193 = arg82_1 = None
        unsqueeze_146: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_56, 4);  add_56 = None
        permute_238: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_146, [1, 2, 0, 4, 3]);  unsqueeze_146 = None
        permute_240: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_238, [0, 1, 2, 4, 3]);  permute_238 = None
        view_210: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_240, [256, 512, 64]);  permute_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_163: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_140: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_163, 3);  convert_element_type_163 = None
        unsqueeze_141: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 4);  unsqueeze_140 = None
        view_202: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_141, [1, 16384, 1024]);  unsqueeze_141 = None
        squeeze_dim_182: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_202, 0);  view_202 = None
        unsqueeze_142: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg80_1, 3);  arg80_1 = None
        unsqueeze_143: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 4);  unsqueeze_142 = None
        view_203: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_143, [1, 1024, 1024]);  unsqueeze_143 = None
        squeeze_dim_183: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_203, 0);  view_203 = None
        mm_default_91: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_182, squeeze_dim_183);  squeeze_dim_182 = squeeze_dim_183 = None
        unsqueeze_default_91: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_91, 0);  mm_default_91 = None
        view_204: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_91, [1024, 16, 1, 16, 64]);  unsqueeze_default_91 = None
        permute_232: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_204, [0, 1, 3, 4, 2]);  view_204 = None
        view_205: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_232, [1024, 16, 16, 64]);  permute_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_147: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_205, 4);  view_205 = None
        permute_239: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_147, [1, 2, 4, 0, 3]);  unsqueeze_147 = None
        permute_241: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_239, [0, 1, 4, 3, 2]);  permute_239 = None
        view_211: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_241, [256, 64, 1024]);  permute_241 = None
        bmm_45: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
        view_212: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [16, 16, 512, 1, 1024]);  bmm_45 = None
        permute_242: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 1, 2, 4, 3]);  view_212 = None
        view_213: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_242, [16, 16, 512, 1024]);  permute_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_214: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [16, 16, 1024, 512]);  view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_12: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_214, 2, 1, 9223372036854775807);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_215: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_12, [16, 16, 512, 1023]);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_7: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_5: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_215, [None, None, None, iota_7]);  view_215 = iota_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_57: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_209, index_5);  view_209 = index_5 = None
        add_58: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, 0);  add_57 = None

        # No stacktrace found for following nodes
        mul_tensor_72: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_58, 0.125)
        convert_element_type_default_36: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_72, torch.float32);  mul_tensor_72 = None
        eq_tensor_18: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_36, convert_element_type_default_36)
        abs_default_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_36)
        ne_scalar_18: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_18, inf);  abs_default_18 = None
        mul_tensor_75: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_18, ne_scalar_18);  eq_tensor_18 = ne_scalar_18 = None
        logical_not_default_36: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_75);  mul_tensor_75 = None
        any_dims_18: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_36, [3], True);  logical_not_default_36 = None
        logical_not_default_37: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_18);  any_dims_18 = None
        convert_element_type_default_37: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.float32);  add_58 = None
        mul_tensor_73: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_37, 1);  convert_element_type_default_37 = None
        amax_default_36: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_73, [3], True)
        sub_tensor_36: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_73, amax_default_36);  mul_tensor_73 = amax_default_36 = None
        mul_tensor_74: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_36, 0.125);  sub_tensor_36 = None
        amax_default_37: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_36, [3], True)
        sub_tensor_37: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_36, amax_default_37);  convert_element_type_default_36 = amax_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_37, mul_tensor_74, sub_tensor_37);  logical_not_default_37 = mul_tensor_74 = sub_tensor_37 = None
        exp_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_18);  where_self_18 = None
        sum_6: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [3], True)
        div_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_171: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_148: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_171, 4);  convert_element_type_171 = None
        view_216: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_148, [256, 512, 512]);  unsqueeze_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_136: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_156, 3)
        unsqueeze_137: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 4);  unsqueeze_136 = None
        view_198: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_137, [1, 8192, 1024]);  unsqueeze_137 = None
        squeeze_dim_184: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_198, 0);  view_198 = None
        unsqueeze_138: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg79_1, 3);  arg79_1 = None
        unsqueeze_139: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 4);  unsqueeze_138 = None
        view_199: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_139, [1, 1024, 1024]);  unsqueeze_139 = None
        squeeze_dim_185: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_199, 0);  view_199 = None
        mm_default_92: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_184, squeeze_dim_185);  squeeze_dim_184 = squeeze_dim_185 = None
        unsqueeze_default_92: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_92, 0);  mm_default_92 = None
        view_200: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_92, [512, 16, 1, 16, 64]);  unsqueeze_default_92 = None
        permute_227: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 1, 3, 4, 2]);  view_200 = None
        view_201: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_227, [512, 16, 16, 64]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_149: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_201, 4);  view_201 = None
        permute_244: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_149, [4, 1, 2, 3, 0]);  unsqueeze_149 = None
        permute_246: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_244, [1, 2, 4, 3, 0]);  permute_244 = None
        view_217: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_246, [256, 512, 64]);  permute_246 = None
        bmm_46: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_216, view_217);  view_216 = view_217 = None
        view_218: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [16, 16, 512, 1, 64]);  bmm_46 = None
        permute_247: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_218, [2, 0, 1, 4, 3]);  view_218 = None
        view_219: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_247, [512, 16, 16, 64]);  permute_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_150: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_219, 4);  view_219 = None
        permute_248: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_150, [0, 1, 4, 3, 2]);  unsqueeze_150 = None
        permute_250: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 4, 2]);  permute_248 = None
        clone_34: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_220: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [1, 8192, 1024]);  clone_34 = None
        squeeze_dim_180: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_220, 0);  view_220 = None
        unsqueeze_151: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg83_1, 3);  arg83_1 = None
        unsqueeze_152: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 4);  unsqueeze_151 = None
        permute_249: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_152, [3, 4, 0, 2, 1]);  unsqueeze_152 = None
        permute_251: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_249, [3, 4, 2, 0, 1]);  permute_249 = None
        clone_35: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_251, memory_format = torch.contiguous_format);  permute_251 = None
        view_221: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [1, 1024, 1024]);  clone_35 = None
        squeeze_dim_181: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_221, 0);  view_221 = None
        mm_default_90: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_180, squeeze_dim_181);  squeeze_dim_180 = squeeze_dim_181 = None
        unsqueeze_default_90: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_90, 0);  mm_default_90 = None
        view_222: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_90, [512, 16, 1, 1, 1024]);  unsqueeze_default_90 = None
        permute_252: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 1, 4, 2, 3]);  view_222 = None
        view_223: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_252, [512, 16, 1024]);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_59: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_223, convert_element_type_156);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_176: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_176, [2], correction = 0, keepdim = True)
        getitem_20: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_16: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_176, getitem_21);  convert_element_type_176 = getitem_21 = None
        add_60: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_43: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_44: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, arg84_1);  mul_43 = arg84_1 = None
        add_61: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, arg85_1);  mul_44 = arg85_1 = None
        convert_element_type_177: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_224: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_177, [8192, 1024])
        permute_253: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_10: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg87_1, view_224, permute_253);  arg87_1 = view_224 = permute_253 = None
        view_225: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [512, 16, 4096]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_181: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_225, torch.float32);  view_225 = None
        mul_45: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_181, 0.5)
        mul_46: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_181, 0.7071067811865476);  convert_element_type_181 = None
        erf_5: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_46);  mul_46 = None
        add_62: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_47: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, add_62);  mul_45 = add_62 = None
        convert_element_type_182: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_226: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_182, [8192, 4096]);  convert_element_type_182 = None
        permute_254: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_11: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg89_1, view_226, permute_254);  arg89_1 = view_226 = permute_254 = None
        view_227: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [512, 16, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_63: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_227, convert_element_type_177);  view_227 = convert_element_type_177 = None
        convert_element_type_186: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_186, [2], correction = 0, keepdim = True)
        getitem_22: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_186, getitem_23);  convert_element_type_186 = getitem_23 = None
        add_64: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_48: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_49: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, arg90_1);  mul_48 = arg90_1 = None
        add_65: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, arg91_1);  mul_49 = arg91_1 = None
        convert_element_type_187: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_153: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_187, 3)
        unsqueeze_154: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 4);  unsqueeze_153 = None
        view_228: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_154, [1, 8192, 1024]);  unsqueeze_154 = None
        squeeze_dim_178: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_228, 0);  view_228 = None
        unsqueeze_155: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg92_1, 3);  arg92_1 = None
        unsqueeze_156: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 4);  unsqueeze_155 = None
        view_229: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_156, [1, 1024, 1024]);  unsqueeze_156 = None
        squeeze_dim_179: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_229, 0);  view_229 = None
        mm_default_89: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_178, squeeze_dim_179);  squeeze_dim_178 = squeeze_dim_179 = None
        unsqueeze_default_89: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_89, 0);  mm_default_89 = None
        view_230: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_89, [512, 16, 1, 16, 64]);  unsqueeze_default_89 = None
        permute_259: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_230, [0, 1, 3, 4, 2]);  view_230 = None
        view_231: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_259, [512, 16, 16, 64]);  permute_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_66: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_231, arg96_1);  arg96_1 = None
        unsqueeze_169: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_66, 4);  add_66 = None
        permute_275: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_169, [1, 2, 0, 4, 3]);  unsqueeze_169 = None
        permute_277: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_275, [0, 1, 2, 4, 3]);  permute_275 = None
        view_244: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_277, [256, 512, 64]);  permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_157: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_187, 3)
        unsqueeze_158: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 4);  unsqueeze_157 = None
        view_232: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_158, [1, 8192, 1024]);  unsqueeze_158 = None
        squeeze_dim_176: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_232, 0);  view_232 = None
        unsqueeze_159: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg93_1, 3);  arg93_1 = None
        unsqueeze_160: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 4);  unsqueeze_159 = None
        view_233: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_160, [1, 1024, 1024]);  unsqueeze_160 = None
        squeeze_dim_177: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_233, 0);  view_233 = None
        mm_default_88: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_176, squeeze_dim_177);  squeeze_dim_176 = squeeze_dim_177 = None
        unsqueeze_default_88: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_88, 0);  mm_default_88 = None
        view_234: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_88, [512, 16, 1, 16, 64]);  unsqueeze_default_88 = None
        permute_264: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 1, 3, 4, 2]);  view_234 = None
        view_235: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_264, [512, 16, 16, 64]);  permute_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_170: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_235, 4);  view_235 = None
        permute_276: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_170, [1, 2, 4, 0, 3]);  unsqueeze_170 = None
        permute_278: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_276, [0, 1, 4, 3, 2]);  permute_276 = None
        view_245: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_278, [256, 64, 512]);  permute_278 = None
        bmm_52: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
        view_246: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [16, 16, 512, 1, 512]);  bmm_52 = None
        permute_279: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_246, [0, 1, 2, 4, 3]);  view_246 = None
        view_247: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_279, [16, 16, 512, 512]);  permute_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_67: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_231, arg97_1);  view_231 = arg97_1 = None
        unsqueeze_171: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_67, 4);  add_67 = None
        permute_280: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_171, [1, 2, 0, 4, 3]);  unsqueeze_171 = None
        permute_282: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_280, [0, 1, 2, 4, 3]);  permute_280 = None
        view_248: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_282, [256, 512, 64]);  permute_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_194: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_165: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_194, 3);  convert_element_type_194 = None
        unsqueeze_166: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 4);  unsqueeze_165 = None
        view_240: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_166, [1, 16384, 1024]);  unsqueeze_166 = None
        squeeze_dim_172: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_240, 0);  view_240 = None
        unsqueeze_167: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg95_1, 3);  arg95_1 = None
        unsqueeze_168: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 4);  unsqueeze_167 = None
        view_241: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_168, [1, 1024, 1024]);  unsqueeze_168 = None
        squeeze_dim_173: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_241, 0);  view_241 = None
        mm_default_86: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_172, squeeze_dim_173);  squeeze_dim_172 = squeeze_dim_173 = None
        unsqueeze_default_86: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_86, 0);  mm_default_86 = None
        view_242: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_86, [1024, 16, 1, 16, 64]);  unsqueeze_default_86 = None
        permute_274: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_242, [0, 1, 3, 4, 2]);  view_242 = None
        view_243: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_274, [1024, 16, 16, 64]);  permute_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_172: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_243, 4);  view_243 = None
        permute_281: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_172, [1, 2, 4, 0, 3]);  unsqueeze_172 = None
        permute_283: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_281, [0, 1, 4, 3, 2]);  permute_281 = None
        view_249: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_283, [256, 64, 1024]);  permute_283 = None
        bmm_53: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_248, view_249);  view_248 = view_249 = None
        view_250: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [16, 16, 512, 1, 1024]);  bmm_53 = None
        permute_284: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 1, 2, 4, 3]);  view_250 = None
        view_251: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_284, [16, 16, 512, 1024]);  permute_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_252: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [16, 16, 1024, 512]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_14: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_252, 2, 1, 9223372036854775807);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_253: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_14, [16, 16, 512, 1023]);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_8: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_6: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_253, [None, None, None, iota_8]);  view_253 = iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_68: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_247, index_6);  view_247 = index_6 = None
        add_69: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_68, 0);  add_68 = None

        # No stacktrace found for following nodes
        mul_tensor_68: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_69, 0.125)
        convert_element_type_default_34: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_68, torch.float32);  mul_tensor_68 = None
        eq_tensor_17: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_34, convert_element_type_default_34)
        abs_default_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_34)
        ne_scalar_17: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_17, inf);  abs_default_17 = None
        mul_tensor_71: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_17, ne_scalar_17);  eq_tensor_17 = ne_scalar_17 = None
        logical_not_default_34: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_71);  mul_tensor_71 = None
        any_dims_17: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_34, [3], True);  logical_not_default_34 = None
        logical_not_default_35: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_17);  any_dims_17 = None
        convert_element_type_default_35: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None
        mul_tensor_69: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_35, 1);  convert_element_type_default_35 = None
        amax_default_34: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_69, [3], True)
        sub_tensor_34: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_69, amax_default_34);  mul_tensor_69 = amax_default_34 = None
        mul_tensor_70: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_34, 0.125);  sub_tensor_34 = None
        amax_default_35: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_34, [3], True)
        sub_tensor_35: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_34, amax_default_35);  convert_element_type_default_34 = amax_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_35, mul_tensor_70, sub_tensor_35);  logical_not_default_35 = mul_tensor_70 = sub_tensor_35 = None
        exp_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_17);  where_self_17 = None
        sum_7: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [3], True)
        div_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_202: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_173: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_202, 4);  convert_element_type_202 = None
        view_254: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_173, [256, 512, 512]);  unsqueeze_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_161: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_187, 3)
        unsqueeze_162: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 4);  unsqueeze_161 = None
        view_236: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_162, [1, 8192, 1024]);  unsqueeze_162 = None
        squeeze_dim_174: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_236, 0);  view_236 = None
        unsqueeze_163: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg94_1, 3);  arg94_1 = None
        unsqueeze_164: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 4);  unsqueeze_163 = None
        view_237: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_164, [1, 1024, 1024]);  unsqueeze_164 = None
        squeeze_dim_175: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_237, 0);  view_237 = None
        mm_default_87: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_174, squeeze_dim_175);  squeeze_dim_174 = squeeze_dim_175 = None
        unsqueeze_default_87: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_87, 0);  mm_default_87 = None
        view_238: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_87, [512, 16, 1, 16, 64]);  unsqueeze_default_87 = None
        permute_269: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_238, [0, 1, 3, 4, 2]);  view_238 = None
        view_239: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_269, [512, 16, 16, 64]);  permute_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_174: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_239, 4);  view_239 = None
        permute_286: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_174, [4, 1, 2, 3, 0]);  unsqueeze_174 = None
        permute_288: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_286, [1, 2, 4, 3, 0]);  permute_286 = None
        view_255: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_288, [256, 512, 64]);  permute_288 = None
        bmm_54: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
        view_256: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [16, 16, 512, 1, 64]);  bmm_54 = None
        permute_289: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_256, [2, 0, 1, 4, 3]);  view_256 = None
        view_257: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_289, [512, 16, 16, 64]);  permute_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_175: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_257, 4);  view_257 = None
        permute_290: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_175, [0, 1, 4, 3, 2]);  unsqueeze_175 = None
        permute_292: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_290, [0, 1, 3, 4, 2]);  permute_290 = None
        clone_40: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_258: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [1, 8192, 1024]);  clone_40 = None
        squeeze_dim_170: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_258, 0);  view_258 = None
        unsqueeze_176: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg98_1, 3);  arg98_1 = None
        unsqueeze_177: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 4);  unsqueeze_176 = None
        permute_291: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_177, [3, 4, 0, 2, 1]);  unsqueeze_177 = None
        permute_293: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_291, [3, 4, 2, 0, 1]);  permute_291 = None
        clone_41: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_259: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [1, 1024, 1024]);  clone_41 = None
        squeeze_dim_171: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_259, 0);  view_259 = None
        mm_default_85: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_170, squeeze_dim_171);  squeeze_dim_170 = squeeze_dim_171 = None
        unsqueeze_default_85: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_85, 0);  mm_default_85 = None
        view_260: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_85, [512, 16, 1, 1, 1024]);  unsqueeze_default_85 = None
        permute_294: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_260, [0, 1, 4, 2, 3]);  view_260 = None
        view_261: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_294, [512, 16, 1024]);  permute_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_70: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_261, convert_element_type_187);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_207: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.float32);  add_70 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_207, [2], correction = 0, keepdim = True)
        getitem_24: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_19: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_207, getitem_25);  convert_element_type_207 = getitem_25 = None
        add_71: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_51: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
        mul_52: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, arg99_1);  mul_51 = arg99_1 = None
        add_72: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_52, arg100_1);  mul_52 = arg100_1 = None
        convert_element_type_208: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_262: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_208, [8192, 1024])
        permute_295: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_12: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg102_1, view_262, permute_295);  arg102_1 = view_262 = permute_295 = None
        view_263: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [512, 16, 4096]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_212: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        mul_53: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.5)
        mul_54: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.7071067811865476);  convert_element_type_212 = None
        erf_6: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_73: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_55: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, add_73);  mul_53 = add_73 = None
        convert_element_type_213: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_264: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_213, [8192, 4096]);  convert_element_type_213 = None
        permute_296: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_13: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg104_1, view_264, permute_296);  arg104_1 = view_264 = permute_296 = None
        view_265: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [512, 16, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_74: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_265, convert_element_type_208);  view_265 = convert_element_type_208 = None
        convert_element_type_217: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.float32);  add_74 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_217, [2], correction = 0, keepdim = True)
        getitem_26: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_217, getitem_27);  convert_element_type_217 = getitem_27 = None
        add_75: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_56: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_57: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, arg105_1);  mul_56 = arg105_1 = None
        add_76: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, arg106_1);  mul_57 = arg106_1 = None
        convert_element_type_218: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_178: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_218, 3)
        unsqueeze_179: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 4);  unsqueeze_178 = None
        view_266: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_179, [1, 8192, 1024]);  unsqueeze_179 = None
        squeeze_dim_168: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_266, 0);  view_266 = None
        unsqueeze_180: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg107_1, 3);  arg107_1 = None
        unsqueeze_181: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 4);  unsqueeze_180 = None
        view_267: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_181, [1, 1024, 1024]);  unsqueeze_181 = None
        squeeze_dim_169: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_267, 0);  view_267 = None
        mm_default_84: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_168, squeeze_dim_169);  squeeze_dim_168 = squeeze_dim_169 = None
        unsqueeze_default_84: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_84, 0);  mm_default_84 = None
        view_268: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_84, [512, 16, 1, 16, 64]);  unsqueeze_default_84 = None
        permute_301: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_268, [0, 1, 3, 4, 2]);  view_268 = None
        view_269: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_301, [512, 16, 16, 64]);  permute_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_77: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_269, arg111_1);  arg111_1 = None
        unsqueeze_194: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_77, 4);  add_77 = None
        permute_317: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_194, [1, 2, 0, 4, 3]);  unsqueeze_194 = None
        permute_319: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_317, [0, 1, 2, 4, 3]);  permute_317 = None
        view_282: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_319, [256, 512, 64]);  permute_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_182: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_218, 3)
        unsqueeze_183: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 4);  unsqueeze_182 = None
        view_270: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_183, [1, 8192, 1024]);  unsqueeze_183 = None
        squeeze_dim_166: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_270, 0);  view_270 = None
        unsqueeze_184: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg108_1, 3);  arg108_1 = None
        unsqueeze_185: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 4);  unsqueeze_184 = None
        view_271: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_185, [1, 1024, 1024]);  unsqueeze_185 = None
        squeeze_dim_167: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_271, 0);  view_271 = None
        mm_default_83: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_166, squeeze_dim_167);  squeeze_dim_166 = squeeze_dim_167 = None
        unsqueeze_default_83: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_83, 0);  mm_default_83 = None
        view_272: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_83, [512, 16, 1, 16, 64]);  unsqueeze_default_83 = None
        permute_306: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 1, 3, 4, 2]);  view_272 = None
        view_273: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_306, [512, 16, 16, 64]);  permute_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_195: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_273, 4);  view_273 = None
        permute_318: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_195, [1, 2, 4, 0, 3]);  unsqueeze_195 = None
        permute_320: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_318, [0, 1, 4, 3, 2]);  permute_318 = None
        view_283: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_320, [256, 64, 512]);  permute_320 = None
        bmm_60: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_282, view_283);  view_282 = view_283 = None
        view_284: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [16, 16, 512, 1, 512]);  bmm_60 = None
        permute_321: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_284, [0, 1, 2, 4, 3]);  view_284 = None
        view_285: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_321, [16, 16, 512, 512]);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_78: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_269, arg112_1);  view_269 = arg112_1 = None
        unsqueeze_196: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_78, 4);  add_78 = None
        permute_322: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_196, [1, 2, 0, 4, 3]);  unsqueeze_196 = None
        permute_324: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_322, [0, 1, 2, 4, 3]);  permute_322 = None
        view_286: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_324, [256, 512, 64]);  permute_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_225: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_190: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_225, 3);  convert_element_type_225 = None
        unsqueeze_191: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 4);  unsqueeze_190 = None
        view_278: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_191, [1, 16384, 1024]);  unsqueeze_191 = None
        squeeze_dim_162: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_278, 0);  view_278 = None
        unsqueeze_192: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg110_1, 3);  arg110_1 = None
        unsqueeze_193: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 4);  unsqueeze_192 = None
        view_279: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_193, [1, 1024, 1024]);  unsqueeze_193 = None
        squeeze_dim_163: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_279, 0);  view_279 = None
        mm_default_81: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_162, squeeze_dim_163);  squeeze_dim_162 = squeeze_dim_163 = None
        unsqueeze_default_81: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_81, 0);  mm_default_81 = None
        view_280: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_81, [1024, 16, 1, 16, 64]);  unsqueeze_default_81 = None
        permute_316: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_280, [0, 1, 3, 4, 2]);  view_280 = None
        view_281: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_316, [1024, 16, 16, 64]);  permute_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_197: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_281, 4);  view_281 = None
        permute_323: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_197, [1, 2, 4, 0, 3]);  unsqueeze_197 = None
        permute_325: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_323, [0, 1, 4, 3, 2]);  permute_323 = None
        view_287: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_325, [256, 64, 1024]);  permute_325 = None
        bmm_61: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_286, view_287);  view_286 = view_287 = None
        view_288: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [16, 16, 512, 1, 1024]);  bmm_61 = None
        permute_326: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 1, 2, 4, 3]);  view_288 = None
        view_289: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_326, [16, 16, 512, 1024]);  permute_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_290: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_289, [16, 16, 1024, 512]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_16: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_290, 2, 1, 9223372036854775807);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_291: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_16, [16, 16, 512, 1023]);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_9: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_7: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_291, [None, None, None, iota_9]);  view_291 = iota_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_79: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_285, index_7);  view_285 = index_7 = None
        add_80: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_79, 0);  add_79 = None

        # No stacktrace found for following nodes
        mul_tensor_64: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_80, 0.125)
        convert_element_type_default_32: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_64, torch.float32);  mul_tensor_64 = None
        eq_tensor_16: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_32, convert_element_type_default_32)
        abs_default_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_32)
        ne_scalar_16: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_16, inf);  abs_default_16 = None
        mul_tensor_67: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_16, ne_scalar_16);  eq_tensor_16 = ne_scalar_16 = None
        logical_not_default_32: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_67);  mul_tensor_67 = None
        any_dims_16: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_32, [3], True);  logical_not_default_32 = None
        logical_not_default_33: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_16);  any_dims_16 = None
        convert_element_type_default_33: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.float32);  add_80 = None
        mul_tensor_65: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_33, 1);  convert_element_type_default_33 = None
        amax_default_32: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_65, [3], True)
        sub_tensor_32: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_65, amax_default_32);  mul_tensor_65 = amax_default_32 = None
        mul_tensor_66: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_32, 0.125);  sub_tensor_32 = None
        amax_default_33: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_32, [3], True)
        sub_tensor_33: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_32, amax_default_33);  convert_element_type_default_32 = amax_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_33, mul_tensor_66, sub_tensor_33);  logical_not_default_33 = mul_tensor_66 = sub_tensor_33 = None
        exp_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_16);  where_self_16 = None
        sum_8: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [3], True)
        div_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_233: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_198: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_233, 4);  convert_element_type_233 = None
        view_292: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_198, [256, 512, 512]);  unsqueeze_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_186: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_218, 3)
        unsqueeze_187: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 4);  unsqueeze_186 = None
        view_274: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_187, [1, 8192, 1024]);  unsqueeze_187 = None
        squeeze_dim_164: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_274, 0);  view_274 = None
        unsqueeze_188: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg109_1, 3);  arg109_1 = None
        unsqueeze_189: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 4);  unsqueeze_188 = None
        view_275: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_189, [1, 1024, 1024]);  unsqueeze_189 = None
        squeeze_dim_165: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_275, 0);  view_275 = None
        mm_default_82: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_164, squeeze_dim_165);  squeeze_dim_164 = squeeze_dim_165 = None
        unsqueeze_default_82: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_82, 0);  mm_default_82 = None
        view_276: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_82, [512, 16, 1, 16, 64]);  unsqueeze_default_82 = None
        permute_311: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_276, [0, 1, 3, 4, 2]);  view_276 = None
        view_277: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_311, [512, 16, 16, 64]);  permute_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_199: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_277, 4);  view_277 = None
        permute_328: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_199, [4, 1, 2, 3, 0]);  unsqueeze_199 = None
        permute_330: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_328, [1, 2, 4, 3, 0]);  permute_328 = None
        view_293: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_330, [256, 512, 64]);  permute_330 = None
        bmm_62: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_292, view_293);  view_292 = view_293 = None
        view_294: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [16, 16, 512, 1, 64]);  bmm_62 = None
        permute_331: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_294, [2, 0, 1, 4, 3]);  view_294 = None
        view_295: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_331, [512, 16, 16, 64]);  permute_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_200: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_295, 4);  view_295 = None
        permute_332: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 4, 3, 2]);  unsqueeze_200 = None
        permute_334: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_332, [0, 1, 3, 4, 2]);  permute_332 = None
        clone_46: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
        view_296: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [1, 8192, 1024]);  clone_46 = None
        squeeze_dim_160: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_296, 0);  view_296 = None
        unsqueeze_201: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg113_1, 3);  arg113_1 = None
        unsqueeze_202: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 4);  unsqueeze_201 = None
        permute_333: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_202, [3, 4, 0, 2, 1]);  unsqueeze_202 = None
        permute_335: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_333, [3, 4, 2, 0, 1]);  permute_333 = None
        clone_47: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_297: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [1, 1024, 1024]);  clone_47 = None
        squeeze_dim_161: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_297, 0);  view_297 = None
        mm_default_80: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_160, squeeze_dim_161);  squeeze_dim_160 = squeeze_dim_161 = None
        unsqueeze_default_80: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_80, 0);  mm_default_80 = None
        view_298: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_80, [512, 16, 1, 1, 1024]);  unsqueeze_default_80 = None
        permute_336: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_298, [0, 1, 4, 2, 3]);  view_298 = None
        view_299: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_336, [512, 16, 1024]);  permute_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_81: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_299, convert_element_type_218);  view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_238: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_238, [2], correction = 0, keepdim = True)
        getitem_28: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_22: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_238, getitem_29);  convert_element_type_238 = getitem_29 = None
        add_82: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_59: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
        mul_60: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, arg114_1);  mul_59 = arg114_1 = None
        add_83: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, arg115_1);  mul_60 = arg115_1 = None
        convert_element_type_239: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_300: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_239, [8192, 1024])
        permute_337: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_14: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg117_1, view_300, permute_337);  arg117_1 = view_300 = permute_337 = None
        view_301: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [512, 16, 4096]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_243: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_301, torch.float32);  view_301 = None
        mul_61: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_243, 0.5)
        mul_62: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_243, 0.7071067811865476);  convert_element_type_243 = None
        erf_7: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_84: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_63: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, add_84);  mul_61 = add_84 = None
        convert_element_type_244: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_302: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_244, [8192, 4096]);  convert_element_type_244 = None
        permute_338: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_15: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg119_1, view_302, permute_338);  arg119_1 = view_302 = permute_338 = None
        view_303: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [512, 16, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_85: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_303, convert_element_type_239);  view_303 = convert_element_type_239 = None
        convert_element_type_248: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_248, [2], correction = 0, keepdim = True)
        getitem_30: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_248, getitem_31);  convert_element_type_248 = getitem_31 = None
        add_86: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_64: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_65: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, arg120_1);  mul_64 = arg120_1 = None
        add_87: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, arg121_1);  mul_65 = arg121_1 = None
        convert_element_type_249: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_203: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_249, 3)
        unsqueeze_204: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 4);  unsqueeze_203 = None
        view_304: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_204, [1, 8192, 1024]);  unsqueeze_204 = None
        squeeze_dim_158: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_304, 0);  view_304 = None
        unsqueeze_205: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg122_1, 3);  arg122_1 = None
        unsqueeze_206: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 4);  unsqueeze_205 = None
        view_305: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_206, [1, 1024, 1024]);  unsqueeze_206 = None
        squeeze_dim_159: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_305, 0);  view_305 = None
        mm_default_79: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_158, squeeze_dim_159);  squeeze_dim_158 = squeeze_dim_159 = None
        unsqueeze_default_79: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_79, 0);  mm_default_79 = None
        view_306: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_79, [512, 16, 1, 16, 64]);  unsqueeze_default_79 = None
        permute_343: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_306, [0, 1, 3, 4, 2]);  view_306 = None
        view_307: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_343, [512, 16, 16, 64]);  permute_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_88: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_307, arg126_1);  arg126_1 = None
        unsqueeze_219: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_88, 4);  add_88 = None
        permute_359: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_219, [1, 2, 0, 4, 3]);  unsqueeze_219 = None
        permute_361: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_359, [0, 1, 2, 4, 3]);  permute_359 = None
        view_320: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_361, [256, 512, 64]);  permute_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_207: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_249, 3)
        unsqueeze_208: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 4);  unsqueeze_207 = None
        view_308: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_208, [1, 8192, 1024]);  unsqueeze_208 = None
        squeeze_dim_156: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_308, 0);  view_308 = None
        unsqueeze_209: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg123_1, 3);  arg123_1 = None
        unsqueeze_210: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 4);  unsqueeze_209 = None
        view_309: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_210, [1, 1024, 1024]);  unsqueeze_210 = None
        squeeze_dim_157: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_309, 0);  view_309 = None
        mm_default_78: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_156, squeeze_dim_157);  squeeze_dim_156 = squeeze_dim_157 = None
        unsqueeze_default_78: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_78, 0);  mm_default_78 = None
        view_310: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_78, [512, 16, 1, 16, 64]);  unsqueeze_default_78 = None
        permute_348: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_310, [0, 1, 3, 4, 2]);  view_310 = None
        view_311: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_348, [512, 16, 16, 64]);  permute_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_220: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_311, 4);  view_311 = None
        permute_360: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_220, [1, 2, 4, 0, 3]);  unsqueeze_220 = None
        permute_362: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_360, [0, 1, 4, 3, 2]);  permute_360 = None
        view_321: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_362, [256, 64, 512]);  permute_362 = None
        bmm_68: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_320, view_321);  view_320 = view_321 = None
        view_322: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [16, 16, 512, 1, 512]);  bmm_68 = None
        permute_363: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_322, [0, 1, 2, 4, 3]);  view_322 = None
        view_323: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_363, [16, 16, 512, 512]);  permute_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_89: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_307, arg127_1);  view_307 = arg127_1 = None
        unsqueeze_221: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_89, 4);  add_89 = None
        permute_364: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_221, [1, 2, 0, 4, 3]);  unsqueeze_221 = None
        permute_366: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_364, [0, 1, 2, 4, 3]);  permute_364 = None
        view_324: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_366, [256, 512, 64]);  permute_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_256: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_215: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_256, 3);  convert_element_type_256 = None
        unsqueeze_216: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 4);  unsqueeze_215 = None
        view_316: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_216, [1, 16384, 1024]);  unsqueeze_216 = None
        squeeze_dim_152: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_316, 0);  view_316 = None
        unsqueeze_217: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg125_1, 3);  arg125_1 = None
        unsqueeze_218: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 4);  unsqueeze_217 = None
        view_317: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_218, [1, 1024, 1024]);  unsqueeze_218 = None
        squeeze_dim_153: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_317, 0);  view_317 = None
        mm_default_76: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_152, squeeze_dim_153);  squeeze_dim_152 = squeeze_dim_153 = None
        unsqueeze_default_76: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_76, 0);  mm_default_76 = None
        view_318: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_76, [1024, 16, 1, 16, 64]);  unsqueeze_default_76 = None
        permute_358: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_318, [0, 1, 3, 4, 2]);  view_318 = None
        view_319: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_358, [1024, 16, 16, 64]);  permute_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_222: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_319, 4);  view_319 = None
        permute_365: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_222, [1, 2, 4, 0, 3]);  unsqueeze_222 = None
        permute_367: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_365, [0, 1, 4, 3, 2]);  permute_365 = None
        view_325: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_367, [256, 64, 1024]);  permute_367 = None
        bmm_69: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_324, view_325);  view_324 = view_325 = None
        view_326: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [16, 16, 512, 1, 1024]);  bmm_69 = None
        permute_368: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_326, [0, 1, 2, 4, 3]);  view_326 = None
        view_327: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_368, [16, 16, 512, 1024]);  permute_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_328: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_327, [16, 16, 1024, 512]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_18: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_328, 2, 1, 9223372036854775807);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_329: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_18, [16, 16, 512, 1023]);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_10: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_8: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_329, [None, None, None, iota_10]);  view_329 = iota_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_90: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_323, index_8);  view_323 = index_8 = None
        add_91: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_90, 0);  add_90 = None

        # No stacktrace found for following nodes
        mul_tensor_60: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_91, 0.125)
        convert_element_type_default_30: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_60, torch.float32);  mul_tensor_60 = None
        eq_tensor_15: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_30, convert_element_type_default_30)
        abs_default_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_30)
        ne_scalar_15: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_15, inf);  abs_default_15 = None
        mul_tensor_63: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_15, ne_scalar_15);  eq_tensor_15 = ne_scalar_15 = None
        logical_not_default_30: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_63);  mul_tensor_63 = None
        any_dims_15: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_30, [3], True);  logical_not_default_30 = None
        logical_not_default_31: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_15);  any_dims_15 = None
        convert_element_type_default_31: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        mul_tensor_61: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_31, 1);  convert_element_type_default_31 = None
        amax_default_30: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_61, [3], True)
        sub_tensor_30: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_61, amax_default_30);  mul_tensor_61 = amax_default_30 = None
        mul_tensor_62: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_30, 0.125);  sub_tensor_30 = None
        amax_default_31: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_30, [3], True)
        sub_tensor_31: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_30, amax_default_31);  convert_element_type_default_30 = amax_default_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_31, mul_tensor_62, sub_tensor_31);  logical_not_default_31 = mul_tensor_62 = sub_tensor_31 = None
        exp_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_15);  where_self_15 = None
        sum_9: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [3], True)
        div_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_264: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_223: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_264, 4);  convert_element_type_264 = None
        view_330: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_223, [256, 512, 512]);  unsqueeze_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_211: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_249, 3)
        unsqueeze_212: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 4);  unsqueeze_211 = None
        view_312: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_212, [1, 8192, 1024]);  unsqueeze_212 = None
        squeeze_dim_154: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_312, 0);  view_312 = None
        unsqueeze_213: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg124_1, 3);  arg124_1 = None
        unsqueeze_214: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 4);  unsqueeze_213 = None
        view_313: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_214, [1, 1024, 1024]);  unsqueeze_214 = None
        squeeze_dim_155: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_313, 0);  view_313 = None
        mm_default_77: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_154, squeeze_dim_155);  squeeze_dim_154 = squeeze_dim_155 = None
        unsqueeze_default_77: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_77, 0);  mm_default_77 = None
        view_314: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_77, [512, 16, 1, 16, 64]);  unsqueeze_default_77 = None
        permute_353: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_314, [0, 1, 3, 4, 2]);  view_314 = None
        view_315: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_353, [512, 16, 16, 64]);  permute_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_224: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_315, 4);  view_315 = None
        permute_370: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_224, [4, 1, 2, 3, 0]);  unsqueeze_224 = None
        permute_372: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_370, [1, 2, 4, 3, 0]);  permute_370 = None
        view_331: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_372, [256, 512, 64]);  permute_372 = None
        bmm_70: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_330, view_331);  view_330 = view_331 = None
        view_332: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [16, 16, 512, 1, 64]);  bmm_70 = None
        permute_373: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_332, [2, 0, 1, 4, 3]);  view_332 = None
        view_333: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_373, [512, 16, 16, 64]);  permute_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_225: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_333, 4);  view_333 = None
        permute_374: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_225, [0, 1, 4, 3, 2]);  unsqueeze_225 = None
        permute_376: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_374, [0, 1, 3, 4, 2]);  permute_374 = None
        clone_52: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format);  permute_376 = None
        view_334: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [1, 8192, 1024]);  clone_52 = None
        squeeze_dim_150: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_334, 0);  view_334 = None
        unsqueeze_226: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg128_1, 3);  arg128_1 = None
        unsqueeze_227: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 4);  unsqueeze_226 = None
        permute_375: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_227, [3, 4, 0, 2, 1]);  unsqueeze_227 = None
        permute_377: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_375, [3, 4, 2, 0, 1]);  permute_375 = None
        clone_53: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_377, memory_format = torch.contiguous_format);  permute_377 = None
        view_335: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [1, 1024, 1024]);  clone_53 = None
        squeeze_dim_151: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_335, 0);  view_335 = None
        mm_default_75: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_150, squeeze_dim_151);  squeeze_dim_150 = squeeze_dim_151 = None
        unsqueeze_default_75: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_75, 0);  mm_default_75 = None
        view_336: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_75, [512, 16, 1, 1, 1024]);  unsqueeze_default_75 = None
        permute_378: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_336, [0, 1, 4, 2, 3]);  view_336 = None
        view_337: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_378, [512, 16, 1024]);  permute_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_92: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_337, convert_element_type_249);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_269: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_92, torch.float32);  add_92 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_269, [2], correction = 0, keepdim = True)
        getitem_32: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_25: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_269, getitem_33);  convert_element_type_269 = getitem_33 = None
        add_93: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_67: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_68: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, arg129_1);  mul_67 = arg129_1 = None
        add_94: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_68, arg130_1);  mul_68 = arg130_1 = None
        convert_element_type_270: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_338: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_270, [8192, 1024])
        permute_379: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_16: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg132_1, view_338, permute_379);  arg132_1 = view_338 = permute_379 = None
        view_339: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [512, 16, 4096]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_274: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        mul_69: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.5)
        mul_70: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.7071067811865476);  convert_element_type_274 = None
        erf_8: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_70);  mul_70 = None
        add_95: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_71: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_69, add_95);  mul_69 = add_95 = None
        convert_element_type_275: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_340: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_275, [8192, 4096]);  convert_element_type_275 = None
        permute_380: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_17: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg134_1, view_340, permute_380);  arg134_1 = view_340 = permute_380 = None
        view_341: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [512, 16, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_96: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_341, convert_element_type_270);  view_341 = convert_element_type_270 = None
        convert_element_type_279: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.float32);  add_96 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_279, [2], correction = 0, keepdim = True)
        getitem_34: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_279, getitem_35);  convert_element_type_279 = getitem_35 = None
        add_97: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_72: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_73: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, arg135_1);  mul_72 = arg135_1 = None
        add_98: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, arg136_1);  mul_73 = arg136_1 = None
        convert_element_type_280: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16);  add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_228: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_280, 3)
        unsqueeze_229: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 4);  unsqueeze_228 = None
        view_342: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_229, [1, 8192, 1024]);  unsqueeze_229 = None
        squeeze_dim_148: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_342, 0);  view_342 = None
        unsqueeze_230: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg137_1, 3);  arg137_1 = None
        unsqueeze_231: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 4);  unsqueeze_230 = None
        view_343: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_231, [1, 1024, 1024]);  unsqueeze_231 = None
        squeeze_dim_149: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_343, 0);  view_343 = None
        mm_default_74: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_148, squeeze_dim_149);  squeeze_dim_148 = squeeze_dim_149 = None
        unsqueeze_default_74: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_74, 0);  mm_default_74 = None
        view_344: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_74, [512, 16, 1, 16, 64]);  unsqueeze_default_74 = None
        permute_385: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 1, 3, 4, 2]);  view_344 = None
        view_345: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_385, [512, 16, 16, 64]);  permute_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_99: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_345, arg141_1);  arg141_1 = None
        unsqueeze_244: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_99, 4);  add_99 = None
        permute_401: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_244, [1, 2, 0, 4, 3]);  unsqueeze_244 = None
        permute_403: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_401, [0, 1, 2, 4, 3]);  permute_401 = None
        view_358: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_403, [256, 512, 64]);  permute_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_232: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_280, 3)
        unsqueeze_233: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 4);  unsqueeze_232 = None
        view_346: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_233, [1, 8192, 1024]);  unsqueeze_233 = None
        squeeze_dim_146: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_346, 0);  view_346 = None
        unsqueeze_234: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg138_1, 3);  arg138_1 = None
        unsqueeze_235: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 4);  unsqueeze_234 = None
        view_347: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_235, [1, 1024, 1024]);  unsqueeze_235 = None
        squeeze_dim_147: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_347, 0);  view_347 = None
        mm_default_73: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_146, squeeze_dim_147);  squeeze_dim_146 = squeeze_dim_147 = None
        unsqueeze_default_73: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_73, 0);  mm_default_73 = None
        view_348: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_73, [512, 16, 1, 16, 64]);  unsqueeze_default_73 = None
        permute_390: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_348, [0, 1, 3, 4, 2]);  view_348 = None
        view_349: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_390, [512, 16, 16, 64]);  permute_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_245: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_349, 4);  view_349 = None
        permute_402: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_245, [1, 2, 4, 0, 3]);  unsqueeze_245 = None
        permute_404: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_402, [0, 1, 4, 3, 2]);  permute_402 = None
        view_359: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_404, [256, 64, 512]);  permute_404 = None
        bmm_76: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_358, view_359);  view_358 = view_359 = None
        view_360: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_76, [16, 16, 512, 1, 512]);  bmm_76 = None
        permute_405: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_360, [0, 1, 2, 4, 3]);  view_360 = None
        view_361: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_405, [16, 16, 512, 512]);  permute_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_100: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_345, arg142_1);  view_345 = arg142_1 = None
        unsqueeze_246: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_100, 4);  add_100 = None
        permute_406: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_246, [1, 2, 0, 4, 3]);  unsqueeze_246 = None
        permute_408: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_406, [0, 1, 2, 4, 3]);  permute_406 = None
        view_362: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_408, [256, 512, 64]);  permute_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_287: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_240: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_287, 3);  convert_element_type_287 = None
        unsqueeze_241: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 4);  unsqueeze_240 = None
        view_354: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_241, [1, 16384, 1024]);  unsqueeze_241 = None
        squeeze_dim_142: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_354, 0);  view_354 = None
        unsqueeze_242: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg140_1, 3);  arg140_1 = None
        unsqueeze_243: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 4);  unsqueeze_242 = None
        view_355: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_243, [1, 1024, 1024]);  unsqueeze_243 = None
        squeeze_dim_143: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_355, 0);  view_355 = None
        mm_default_71: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_142, squeeze_dim_143);  squeeze_dim_142 = squeeze_dim_143 = None
        unsqueeze_default_71: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_71, 0);  mm_default_71 = None
        view_356: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_71, [1024, 16, 1, 16, 64]);  unsqueeze_default_71 = None
        permute_400: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_356, [0, 1, 3, 4, 2]);  view_356 = None
        view_357: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_400, [1024, 16, 16, 64]);  permute_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_247: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_357, 4);  view_357 = None
        permute_407: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_247, [1, 2, 4, 0, 3]);  unsqueeze_247 = None
        permute_409: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_407, [0, 1, 4, 3, 2]);  permute_407 = None
        view_363: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_409, [256, 64, 1024]);  permute_409 = None
        bmm_77: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_362, view_363);  view_362 = view_363 = None
        view_364: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_77, [16, 16, 512, 1, 1024]);  bmm_77 = None
        permute_410: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_364, [0, 1, 2, 4, 3]);  view_364 = None
        view_365: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_410, [16, 16, 512, 1024]);  permute_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_366: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [16, 16, 1024, 512]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_20: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_366, 2, 1, 9223372036854775807);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_367: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_20, [16, 16, 512, 1023]);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_11: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_9: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_367, [None, None, None, iota_11]);  view_367 = iota_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_101: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_361, index_9);  view_361 = index_9 = None
        add_102: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, 0);  add_101 = None

        # No stacktrace found for following nodes
        mul_tensor_56: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, 0.125)
        convert_element_type_default_28: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_56, torch.float32);  mul_tensor_56 = None
        eq_tensor_14: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_28, convert_element_type_default_28)
        abs_default_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_28)
        ne_scalar_14: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_14, inf);  abs_default_14 = None
        mul_tensor_59: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_14, ne_scalar_14);  eq_tensor_14 = ne_scalar_14 = None
        logical_not_default_28: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_59);  mul_tensor_59 = None
        any_dims_14: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_28, [3], True);  logical_not_default_28 = None
        logical_not_default_29: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_14);  any_dims_14 = None
        convert_element_type_default_29: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.float32);  add_102 = None
        mul_tensor_57: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_29, 1);  convert_element_type_default_29 = None
        amax_default_28: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_57, [3], True)
        sub_tensor_28: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_57, amax_default_28);  mul_tensor_57 = amax_default_28 = None
        mul_tensor_58: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_28, 0.125);  sub_tensor_28 = None
        amax_default_29: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_28, [3], True)
        sub_tensor_29: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_28, amax_default_29);  convert_element_type_default_28 = amax_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_29, mul_tensor_58, sub_tensor_29);  logical_not_default_29 = mul_tensor_58 = sub_tensor_29 = None
        exp_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_14);  where_self_14 = None
        sum_10: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [3], True)
        div_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_295: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_248: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_295, 4);  convert_element_type_295 = None
        view_368: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_248, [256, 512, 512]);  unsqueeze_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_236: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_280, 3)
        unsqueeze_237: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 4);  unsqueeze_236 = None
        view_350: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_237, [1, 8192, 1024]);  unsqueeze_237 = None
        squeeze_dim_144: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_350, 0);  view_350 = None
        unsqueeze_238: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg139_1, 3);  arg139_1 = None
        unsqueeze_239: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 4);  unsqueeze_238 = None
        view_351: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_239, [1, 1024, 1024]);  unsqueeze_239 = None
        squeeze_dim_145: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_351, 0);  view_351 = None
        mm_default_72: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_144, squeeze_dim_145);  squeeze_dim_144 = squeeze_dim_145 = None
        unsqueeze_default_72: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_72, 0);  mm_default_72 = None
        view_352: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_72, [512, 16, 1, 16, 64]);  unsqueeze_default_72 = None
        permute_395: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_352, [0, 1, 3, 4, 2]);  view_352 = None
        view_353: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_395, [512, 16, 16, 64]);  permute_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_249: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_353, 4);  view_353 = None
        permute_412: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_249, [4, 1, 2, 3, 0]);  unsqueeze_249 = None
        permute_414: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_412, [1, 2, 4, 3, 0]);  permute_412 = None
        view_369: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_414, [256, 512, 64]);  permute_414 = None
        bmm_78: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_368, view_369);  view_368 = view_369 = None
        view_370: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_78, [16, 16, 512, 1, 64]);  bmm_78 = None
        permute_415: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_370, [2, 0, 1, 4, 3]);  view_370 = None
        view_371: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_415, [512, 16, 16, 64]);  permute_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_250: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_371, 4);  view_371 = None
        permute_416: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_250, [0, 1, 4, 3, 2]);  unsqueeze_250 = None
        permute_418: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_416, [0, 1, 3, 4, 2]);  permute_416 = None
        clone_58: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_372: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [1, 8192, 1024]);  clone_58 = None
        squeeze_dim_140: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_372, 0);  view_372 = None
        unsqueeze_251: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg143_1, 3);  arg143_1 = None
        unsqueeze_252: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 4);  unsqueeze_251 = None
        permute_417: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_252, [3, 4, 0, 2, 1]);  unsqueeze_252 = None
        permute_419: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_417, [3, 4, 2, 0, 1]);  permute_417 = None
        clone_59: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_373: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [1, 1024, 1024]);  clone_59 = None
        squeeze_dim_141: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_373, 0);  view_373 = None
        mm_default_70: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_140, squeeze_dim_141);  squeeze_dim_140 = squeeze_dim_141 = None
        unsqueeze_default_70: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_70, 0);  mm_default_70 = None
        view_374: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_70, [512, 16, 1, 1, 1024]);  unsqueeze_default_70 = None
        permute_420: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_374, [0, 1, 4, 2, 3]);  view_374 = None
        view_375: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_420, [512, 16, 1024]);  permute_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_103: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_375, convert_element_type_280);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_300: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.float32);  add_103 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_300, [2], correction = 0, keepdim = True)
        getitem_36: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_28: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_300, getitem_37);  convert_element_type_300 = getitem_37 = None
        add_104: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_75: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
        mul_76: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, arg144_1);  mul_75 = arg144_1 = None
        add_105: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, arg145_1);  mul_76 = arg145_1 = None
        convert_element_type_301: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_376: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_301, [8192, 1024])
        permute_421: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_18: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg147_1, view_376, permute_421);  arg147_1 = view_376 = permute_421 = None
        view_377: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [512, 16, 4096]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_305: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None
        mul_77: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_305, 0.5)
        mul_78: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_305, 0.7071067811865476);  convert_element_type_305 = None
        erf_9: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_106: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_79: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, add_106);  mul_77 = add_106 = None
        convert_element_type_306: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_378: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_306, [8192, 4096]);  convert_element_type_306 = None
        permute_422: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_19: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg149_1, view_378, permute_422);  arg149_1 = view_378 = permute_422 = None
        view_379: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [512, 16, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_107: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_379, convert_element_type_301);  view_379 = convert_element_type_301 = None
        convert_element_type_310: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.float32);  add_107 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_310, [2], correction = 0, keepdim = True)
        getitem_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_310, getitem_39);  convert_element_type_310 = getitem_39 = None
        add_108: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_80: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_81: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, arg150_1);  mul_80 = arg150_1 = None
        add_109: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, arg151_1);  mul_81 = arg151_1 = None
        convert_element_type_311: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_253: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_311, 3)
        unsqueeze_254: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 4);  unsqueeze_253 = None
        view_380: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_254, [1, 8192, 1024]);  unsqueeze_254 = None
        squeeze_dim_138: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_380, 0);  view_380 = None
        unsqueeze_255: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg152_1, 3);  arg152_1 = None
        unsqueeze_256: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 4);  unsqueeze_255 = None
        view_381: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_256, [1, 1024, 1024]);  unsqueeze_256 = None
        squeeze_dim_139: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_381, 0);  view_381 = None
        mm_default_69: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_138, squeeze_dim_139);  squeeze_dim_138 = squeeze_dim_139 = None
        unsqueeze_default_69: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_69, 0);  mm_default_69 = None
        view_382: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_69, [512, 16, 1, 16, 64]);  unsqueeze_default_69 = None
        permute_427: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_382, [0, 1, 3, 4, 2]);  view_382 = None
        view_383: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_427, [512, 16, 16, 64]);  permute_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_110: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_383, arg156_1);  arg156_1 = None
        unsqueeze_269: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_110, 4);  add_110 = None
        permute_443: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_269, [1, 2, 0, 4, 3]);  unsqueeze_269 = None
        permute_445: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_443, [0, 1, 2, 4, 3]);  permute_443 = None
        view_396: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_445, [256, 512, 64]);  permute_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_257: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_311, 3)
        unsqueeze_258: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 4);  unsqueeze_257 = None
        view_384: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_258, [1, 8192, 1024]);  unsqueeze_258 = None
        squeeze_dim_136: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_384, 0);  view_384 = None
        unsqueeze_259: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg153_1, 3);  arg153_1 = None
        unsqueeze_260: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 4);  unsqueeze_259 = None
        view_385: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_260, [1, 1024, 1024]);  unsqueeze_260 = None
        squeeze_dim_137: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_385, 0);  view_385 = None
        mm_default_68: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_136, squeeze_dim_137);  squeeze_dim_136 = squeeze_dim_137 = None
        unsqueeze_default_68: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_68, 0);  mm_default_68 = None
        view_386: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_68, [512, 16, 1, 16, 64]);  unsqueeze_default_68 = None
        permute_432: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_386, [0, 1, 3, 4, 2]);  view_386 = None
        view_387: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_432, [512, 16, 16, 64]);  permute_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_270: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_387, 4);  view_387 = None
        permute_444: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_270, [1, 2, 4, 0, 3]);  unsqueeze_270 = None
        permute_446: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_444, [0, 1, 4, 3, 2]);  permute_444 = None
        view_397: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_446, [256, 64, 512]);  permute_446 = None
        bmm_84: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_396, view_397);  view_396 = view_397 = None
        view_398: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_84, [16, 16, 512, 1, 512]);  bmm_84 = None
        permute_447: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 1, 2, 4, 3]);  view_398 = None
        view_399: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_447, [16, 16, 512, 512]);  permute_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_111: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_383, arg157_1);  view_383 = arg157_1 = None
        unsqueeze_271: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_111, 4);  add_111 = None
        permute_448: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_271, [1, 2, 0, 4, 3]);  unsqueeze_271 = None
        permute_450: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_448, [0, 1, 2, 4, 3]);  permute_448 = None
        view_400: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_450, [256, 512, 64]);  permute_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_318: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_265: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_318, 3);  convert_element_type_318 = None
        unsqueeze_266: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 4);  unsqueeze_265 = None
        view_392: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_266, [1, 16384, 1024]);  unsqueeze_266 = None
        squeeze_dim_132: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_392, 0);  view_392 = None
        unsqueeze_267: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg155_1, 3);  arg155_1 = None
        unsqueeze_268: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 4);  unsqueeze_267 = None
        view_393: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_268, [1, 1024, 1024]);  unsqueeze_268 = None
        squeeze_dim_133: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_393, 0);  view_393 = None
        mm_default_66: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_132, squeeze_dim_133);  squeeze_dim_132 = squeeze_dim_133 = None
        unsqueeze_default_66: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_66, 0);  mm_default_66 = None
        view_394: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_66, [1024, 16, 1, 16, 64]);  unsqueeze_default_66 = None
        permute_442: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_394, [0, 1, 3, 4, 2]);  view_394 = None
        view_395: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_442, [1024, 16, 16, 64]);  permute_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_272: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_395, 4);  view_395 = None
        permute_449: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_272, [1, 2, 4, 0, 3]);  unsqueeze_272 = None
        permute_451: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_449, [0, 1, 4, 3, 2]);  permute_449 = None
        view_401: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_451, [256, 64, 1024]);  permute_451 = None
        bmm_85: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_400, view_401);  view_400 = view_401 = None
        view_402: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_85, [16, 16, 512, 1, 1024]);  bmm_85 = None
        permute_452: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 1, 2, 4, 3]);  view_402 = None
        view_403: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_452, [16, 16, 512, 1024]);  permute_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_404: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_403, [16, 16, 1024, 512]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_22: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_404, 2, 1, 9223372036854775807);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_405: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_22, [16, 16, 512, 1023]);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_12: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_10: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_405, [None, None, None, iota_12]);  view_405 = iota_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_112: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_399, index_10);  view_399 = index_10 = None
        add_113: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, 0);  add_112 = None

        # No stacktrace found for following nodes
        mul_tensor_52: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, 0.125)
        convert_element_type_default_26: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_52, torch.float32);  mul_tensor_52 = None
        eq_tensor_13: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_26, convert_element_type_default_26)
        abs_default_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_26)
        ne_scalar_13: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_13, inf);  abs_default_13 = None
        mul_tensor_55: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_13, ne_scalar_13);  eq_tensor_13 = ne_scalar_13 = None
        logical_not_default_26: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_55);  mul_tensor_55 = None
        any_dims_13: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_26, [3], True);  logical_not_default_26 = None
        logical_not_default_27: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_13);  any_dims_13 = None
        convert_element_type_default_27: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.float32);  add_113 = None
        mul_tensor_53: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_27, 1);  convert_element_type_default_27 = None
        amax_default_26: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_53, [3], True)
        sub_tensor_26: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_53, amax_default_26);  mul_tensor_53 = amax_default_26 = None
        mul_tensor_54: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_26, 0.125);  sub_tensor_26 = None
        amax_default_27: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_26, [3], True)
        sub_tensor_27: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_26, amax_default_27);  convert_element_type_default_26 = amax_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_27, mul_tensor_54, sub_tensor_27);  logical_not_default_27 = mul_tensor_54 = sub_tensor_27 = None
        exp_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_13);  where_self_13 = None
        sum_11: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [3], True)
        div_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_326: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_273: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_326, 4);  convert_element_type_326 = None
        view_406: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_273, [256, 512, 512]);  unsqueeze_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_261: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_311, 3)
        unsqueeze_262: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 4);  unsqueeze_261 = None
        view_388: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_262, [1, 8192, 1024]);  unsqueeze_262 = None
        squeeze_dim_134: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_388, 0);  view_388 = None
        unsqueeze_263: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg154_1, 3);  arg154_1 = None
        unsqueeze_264: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 4);  unsqueeze_263 = None
        view_389: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_264, [1, 1024, 1024]);  unsqueeze_264 = None
        squeeze_dim_135: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_389, 0);  view_389 = None
        mm_default_67: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_134, squeeze_dim_135);  squeeze_dim_134 = squeeze_dim_135 = None
        unsqueeze_default_67: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_67, 0);  mm_default_67 = None
        view_390: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_67, [512, 16, 1, 16, 64]);  unsqueeze_default_67 = None
        permute_437: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_390, [0, 1, 3, 4, 2]);  view_390 = None
        view_391: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_437, [512, 16, 16, 64]);  permute_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_274: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_391, 4);  view_391 = None
        permute_454: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_274, [4, 1, 2, 3, 0]);  unsqueeze_274 = None
        permute_456: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_454, [1, 2, 4, 3, 0]);  permute_454 = None
        view_407: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_456, [256, 512, 64]);  permute_456 = None
        bmm_86: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_406, view_407);  view_406 = view_407 = None
        view_408: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_86, [16, 16, 512, 1, 64]);  bmm_86 = None
        permute_457: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_408, [2, 0, 1, 4, 3]);  view_408 = None
        view_409: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_457, [512, 16, 16, 64]);  permute_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_275: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_409, 4);  view_409 = None
        permute_458: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_275, [0, 1, 4, 3, 2]);  unsqueeze_275 = None
        permute_460: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_458, [0, 1, 3, 4, 2]);  permute_458 = None
        clone_64: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
        view_410: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [1, 8192, 1024]);  clone_64 = None
        squeeze_dim_130: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_410, 0);  view_410 = None
        unsqueeze_276: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg158_1, 3);  arg158_1 = None
        unsqueeze_277: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 4);  unsqueeze_276 = None
        permute_459: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_277, [3, 4, 0, 2, 1]);  unsqueeze_277 = None
        permute_461: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_459, [3, 4, 2, 0, 1]);  permute_459 = None
        clone_65: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
        view_411: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1, 1024, 1024]);  clone_65 = None
        squeeze_dim_131: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_411, 0);  view_411 = None
        mm_default_65: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_130, squeeze_dim_131);  squeeze_dim_130 = squeeze_dim_131 = None
        unsqueeze_default_65: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_65, 0);  mm_default_65 = None
        view_412: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_65, [512, 16, 1, 1, 1024]);  unsqueeze_default_65 = None
        permute_462: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 1, 4, 2, 3]);  view_412 = None
        view_413: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_462, [512, 16, 1024]);  permute_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_114: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_413, convert_element_type_311);  view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_331: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_331, [2], correction = 0, keepdim = True)
        getitem_40: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_31: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, getitem_41);  convert_element_type_331 = getitem_41 = None
        add_115: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_83: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
        mul_84: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, arg159_1);  mul_83 = arg159_1 = None
        add_116: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_84, arg160_1);  mul_84 = arg160_1 = None
        convert_element_type_332: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_414: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_332, [8192, 1024])
        permute_463: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_20: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg162_1, view_414, permute_463);  arg162_1 = view_414 = permute_463 = None
        view_415: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [512, 16, 4096]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_336: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_85: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_336, 0.5)
        mul_86: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_336, 0.7071067811865476);  convert_element_type_336 = None
        erf_10: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_117: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_87: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, add_117);  mul_85 = add_117 = None
        convert_element_type_337: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_416: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [8192, 4096]);  convert_element_type_337 = None
        permute_464: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_21: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg164_1, view_416, permute_464);  arg164_1 = view_416 = permute_464 = None
        view_417: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [512, 16, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_118: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_417, convert_element_type_332);  view_417 = convert_element_type_332 = None
        convert_element_type_341: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_118, torch.float32);  add_118 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_341, [2], correction = 0, keepdim = True)
        getitem_42: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_341, getitem_43);  convert_element_type_341 = getitem_43 = None
        add_119: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_88: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_89: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, arg165_1);  mul_88 = arg165_1 = None
        add_120: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_89, arg166_1);  mul_89 = arg166_1 = None
        convert_element_type_342: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_278: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_342, 3)
        unsqueeze_279: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 4);  unsqueeze_278 = None
        view_418: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_279, [1, 8192, 1024]);  unsqueeze_279 = None
        squeeze_dim_128: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_418, 0);  view_418 = None
        unsqueeze_280: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg167_1, 3);  arg167_1 = None
        unsqueeze_281: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 4);  unsqueeze_280 = None
        view_419: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_281, [1, 1024, 1024]);  unsqueeze_281 = None
        squeeze_dim_129: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_419, 0);  view_419 = None
        mm_default_64: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_128, squeeze_dim_129);  squeeze_dim_128 = squeeze_dim_129 = None
        unsqueeze_default_64: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_64, 0);  mm_default_64 = None
        view_420: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_64, [512, 16, 1, 16, 64]);  unsqueeze_default_64 = None
        permute_469: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 1, 3, 4, 2]);  view_420 = None
        view_421: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_469, [512, 16, 16, 64]);  permute_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_121: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_421, arg171_1);  arg171_1 = None
        unsqueeze_294: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_121, 4);  add_121 = None
        permute_485: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_294, [1, 2, 0, 4, 3]);  unsqueeze_294 = None
        permute_487: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_485, [0, 1, 2, 4, 3]);  permute_485 = None
        view_434: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_487, [256, 512, 64]);  permute_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_282: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_342, 3)
        unsqueeze_283: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 4);  unsqueeze_282 = None
        view_422: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_283, [1, 8192, 1024]);  unsqueeze_283 = None
        squeeze_dim_126: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_422, 0);  view_422 = None
        unsqueeze_284: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg168_1, 3);  arg168_1 = None
        unsqueeze_285: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 4);  unsqueeze_284 = None
        view_423: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_285, [1, 1024, 1024]);  unsqueeze_285 = None
        squeeze_dim_127: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_423, 0);  view_423 = None
        mm_default_63: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_126, squeeze_dim_127);  squeeze_dim_126 = squeeze_dim_127 = None
        unsqueeze_default_63: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_63, 0);  mm_default_63 = None
        view_424: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_63, [512, 16, 1, 16, 64]);  unsqueeze_default_63 = None
        permute_474: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_424, [0, 1, 3, 4, 2]);  view_424 = None
        view_425: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_474, [512, 16, 16, 64]);  permute_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_295: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_425, 4);  view_425 = None
        permute_486: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_295, [1, 2, 4, 0, 3]);  unsqueeze_295 = None
        permute_488: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_486, [0, 1, 4, 3, 2]);  permute_486 = None
        view_435: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_488, [256, 64, 512]);  permute_488 = None
        bmm_92: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_434, view_435);  view_434 = view_435 = None
        view_436: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_92, [16, 16, 512, 1, 512]);  bmm_92 = None
        permute_489: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_436, [0, 1, 2, 4, 3]);  view_436 = None
        view_437: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_489, [16, 16, 512, 512]);  permute_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_122: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_421, arg172_1);  view_421 = arg172_1 = None
        unsqueeze_296: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_122, 4);  add_122 = None
        permute_490: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_296, [1, 2, 0, 4, 3]);  unsqueeze_296 = None
        permute_492: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_490, [0, 1, 2, 4, 3]);  permute_490 = None
        view_438: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_492, [256, 512, 64]);  permute_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_349: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_290: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_349, 3);  convert_element_type_349 = None
        unsqueeze_291: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 4);  unsqueeze_290 = None
        view_430: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_291, [1, 16384, 1024]);  unsqueeze_291 = None
        squeeze_dim_122: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_430, 0);  view_430 = None
        unsqueeze_292: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg170_1, 3);  arg170_1 = None
        unsqueeze_293: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 4);  unsqueeze_292 = None
        view_431: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_293, [1, 1024, 1024]);  unsqueeze_293 = None
        squeeze_dim_123: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_431, 0);  view_431 = None
        mm_default_61: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_122, squeeze_dim_123);  squeeze_dim_122 = squeeze_dim_123 = None
        unsqueeze_default_61: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_61, 0);  mm_default_61 = None
        view_432: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_61, [1024, 16, 1, 16, 64]);  unsqueeze_default_61 = None
        permute_484: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 1, 3, 4, 2]);  view_432 = None
        view_433: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_484, [1024, 16, 16, 64]);  permute_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_297: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_433, 4);  view_433 = None
        permute_491: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_297, [1, 2, 4, 0, 3]);  unsqueeze_297 = None
        permute_493: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_491, [0, 1, 4, 3, 2]);  permute_491 = None
        view_439: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_493, [256, 64, 1024]);  permute_493 = None
        bmm_93: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_438, view_439);  view_438 = view_439 = None
        view_440: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_93, [16, 16, 512, 1, 1024]);  bmm_93 = None
        permute_494: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_440, [0, 1, 2, 4, 3]);  view_440 = None
        view_441: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_494, [16, 16, 512, 1024]);  permute_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_442: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [16, 16, 1024, 512]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_24: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_442, 2, 1, 9223372036854775807);  view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_443: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_24, [16, 16, 512, 1023]);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_13: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_11: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_443, [None, None, None, iota_13]);  view_443 = iota_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_123: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_437, index_11);  view_437 = index_11 = None
        add_124: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, 0);  add_123 = None

        # No stacktrace found for following nodes
        mul_tensor_48: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_124, 0.125)
        convert_element_type_default_24: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_48, torch.float32);  mul_tensor_48 = None
        eq_tensor_12: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_24, convert_element_type_default_24)
        abs_default_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_24)
        ne_scalar_12: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_12, inf);  abs_default_12 = None
        mul_tensor_51: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_12, ne_scalar_12);  eq_tensor_12 = ne_scalar_12 = None
        logical_not_default_24: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_51);  mul_tensor_51 = None
        any_dims_12: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_24, [3], True);  logical_not_default_24 = None
        logical_not_default_25: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_12);  any_dims_12 = None
        convert_element_type_default_25: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.float32);  add_124 = None
        mul_tensor_49: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_25, 1);  convert_element_type_default_25 = None
        amax_default_24: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_49, [3], True)
        sub_tensor_24: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_49, amax_default_24);  mul_tensor_49 = amax_default_24 = None
        mul_tensor_50: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_24, 0.125);  sub_tensor_24 = None
        amax_default_25: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_24, [3], True)
        sub_tensor_25: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_24, amax_default_25);  convert_element_type_default_24 = amax_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_25, mul_tensor_50, sub_tensor_25);  logical_not_default_25 = mul_tensor_50 = sub_tensor_25 = None
        exp_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_12);  where_self_12 = None
        sum_12: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [3], True)
        div_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_357: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_298: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_357, 4);  convert_element_type_357 = None
        view_444: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_298, [256, 512, 512]);  unsqueeze_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_286: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_342, 3)
        unsqueeze_287: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 4);  unsqueeze_286 = None
        view_426: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_287, [1, 8192, 1024]);  unsqueeze_287 = None
        squeeze_dim_124: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_426, 0);  view_426 = None
        unsqueeze_288: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg169_1, 3);  arg169_1 = None
        unsqueeze_289: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 4);  unsqueeze_288 = None
        view_427: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_289, [1, 1024, 1024]);  unsqueeze_289 = None
        squeeze_dim_125: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_427, 0);  view_427 = None
        mm_default_62: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_124, squeeze_dim_125);  squeeze_dim_124 = squeeze_dim_125 = None
        unsqueeze_default_62: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_62, 0);  mm_default_62 = None
        view_428: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_62, [512, 16, 1, 16, 64]);  unsqueeze_default_62 = None
        permute_479: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_428, [0, 1, 3, 4, 2]);  view_428 = None
        view_429: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_479, [512, 16, 16, 64]);  permute_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_299: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_429, 4);  view_429 = None
        permute_496: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_299, [4, 1, 2, 3, 0]);  unsqueeze_299 = None
        permute_498: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_496, [1, 2, 4, 3, 0]);  permute_496 = None
        view_445: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_498, [256, 512, 64]);  permute_498 = None
        bmm_94: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_444, view_445);  view_444 = view_445 = None
        view_446: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_94, [16, 16, 512, 1, 64]);  bmm_94 = None
        permute_499: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_446, [2, 0, 1, 4, 3]);  view_446 = None
        view_447: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_499, [512, 16, 16, 64]);  permute_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_300: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_447, 4);  view_447 = None
        permute_500: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_300, [0, 1, 4, 3, 2]);  unsqueeze_300 = None
        permute_502: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_500, [0, 1, 3, 4, 2]);  permute_500 = None
        clone_70: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
        view_448: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [1, 8192, 1024]);  clone_70 = None
        squeeze_dim_120: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_448, 0);  view_448 = None
        unsqueeze_301: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg173_1, 3);  arg173_1 = None
        unsqueeze_302: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 4);  unsqueeze_301 = None
        permute_501: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_302, [3, 4, 0, 2, 1]);  unsqueeze_302 = None
        permute_503: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_501, [3, 4, 2, 0, 1]);  permute_501 = None
        clone_71: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_503, memory_format = torch.contiguous_format);  permute_503 = None
        view_449: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [1, 1024, 1024]);  clone_71 = None
        squeeze_dim_121: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_449, 0);  view_449 = None
        mm_default_60: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_120, squeeze_dim_121);  squeeze_dim_120 = squeeze_dim_121 = None
        unsqueeze_default_60: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_60, 0);  mm_default_60 = None
        view_450: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_60, [512, 16, 1, 1, 1024]);  unsqueeze_default_60 = None
        permute_504: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_450, [0, 1, 4, 2, 3]);  view_450 = None
        view_451: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_504, [512, 16, 1024]);  permute_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_125: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_451, convert_element_type_342);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_362: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.float32);  add_125 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_362, [2], correction = 0, keepdim = True)
        getitem_44: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_34: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_362, getitem_45);  convert_element_type_362 = getitem_45 = None
        add_126: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_91: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_92: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, arg174_1);  mul_91 = arg174_1 = None
        add_127: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, arg175_1);  mul_92 = arg175_1 = None
        convert_element_type_363: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_452: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_363, [8192, 1024])
        permute_505: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_22: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg177_1, view_452, permute_505);  arg177_1 = view_452 = permute_505 = None
        view_453: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [512, 16, 4096]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_367: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_453, torch.float32);  view_453 = None
        mul_93: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_367, 0.5)
        mul_94: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_367, 0.7071067811865476);  convert_element_type_367 = None
        erf_11: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
        add_128: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_95: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, add_128);  mul_93 = add_128 = None
        convert_element_type_368: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_95, torch.bfloat16);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_454: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_368, [8192, 4096]);  convert_element_type_368 = None
        permute_506: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_23: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg179_1, view_454, permute_506);  arg179_1 = view_454 = permute_506 = None
        view_455: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [512, 16, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_129: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_455, convert_element_type_363);  view_455 = convert_element_type_363 = None
        convert_element_type_372: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.float32);  add_129 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_372, [2], correction = 0, keepdim = True)
        getitem_46: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_372, getitem_47);  convert_element_type_372 = getitem_47 = None
        add_130: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_96: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_97: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, arg180_1);  mul_96 = arg180_1 = None
        add_131: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, arg181_1);  mul_97 = arg181_1 = None
        convert_element_type_373: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_303: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_373, 3)
        unsqueeze_304: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 4);  unsqueeze_303 = None
        view_456: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_304, [1, 8192, 1024]);  unsqueeze_304 = None
        squeeze_dim_118: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_456, 0);  view_456 = None
        unsqueeze_305: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg182_1, 3);  arg182_1 = None
        unsqueeze_306: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 4);  unsqueeze_305 = None
        view_457: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_306, [1, 1024, 1024]);  unsqueeze_306 = None
        squeeze_dim_119: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_457, 0);  view_457 = None
        mm_default_59: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_118, squeeze_dim_119);  squeeze_dim_118 = squeeze_dim_119 = None
        unsqueeze_default_59: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_59, 0);  mm_default_59 = None
        view_458: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_59, [512, 16, 1, 16, 64]);  unsqueeze_default_59 = None
        permute_511: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 4, 2]);  view_458 = None
        view_459: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_511, [512, 16, 16, 64]);  permute_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_132: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_459, arg186_1);  arg186_1 = None
        unsqueeze_319: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_132, 4);  add_132 = None
        permute_527: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_319, [1, 2, 0, 4, 3]);  unsqueeze_319 = None
        permute_529: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_527, [0, 1, 2, 4, 3]);  permute_527 = None
        view_472: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_529, [256, 512, 64]);  permute_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_307: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_373, 3)
        unsqueeze_308: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 4);  unsqueeze_307 = None
        view_460: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_308, [1, 8192, 1024]);  unsqueeze_308 = None
        squeeze_dim_116: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_460, 0);  view_460 = None
        unsqueeze_309: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg183_1, 3);  arg183_1 = None
        unsqueeze_310: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 4);  unsqueeze_309 = None
        view_461: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_310, [1, 1024, 1024]);  unsqueeze_310 = None
        squeeze_dim_117: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_461, 0);  view_461 = None
        mm_default_58: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_116, squeeze_dim_117);  squeeze_dim_116 = squeeze_dim_117 = None
        unsqueeze_default_58: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_58, 0);  mm_default_58 = None
        view_462: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_58, [512, 16, 1, 16, 64]);  unsqueeze_default_58 = None
        permute_516: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_462, [0, 1, 3, 4, 2]);  view_462 = None
        view_463: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_516, [512, 16, 16, 64]);  permute_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_320: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_463, 4);  view_463 = None
        permute_528: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_320, [1, 2, 4, 0, 3]);  unsqueeze_320 = None
        permute_530: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_528, [0, 1, 4, 3, 2]);  permute_528 = None
        view_473: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_530, [256, 64, 512]);  permute_530 = None
        bmm_100: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_472, view_473);  view_472 = view_473 = None
        view_474: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_100, [16, 16, 512, 1, 512]);  bmm_100 = None
        permute_531: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_474, [0, 1, 2, 4, 3]);  view_474 = None
        view_475: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_531, [16, 16, 512, 512]);  permute_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_133: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_459, arg187_1);  view_459 = arg187_1 = None
        unsqueeze_321: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_133, 4);  add_133 = None
        permute_532: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_321, [1, 2, 0, 4, 3]);  unsqueeze_321 = None
        permute_534: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_532, [0, 1, 2, 4, 3]);  permute_532 = None
        view_476: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_534, [256, 512, 64]);  permute_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_380: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_315: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_380, 3);  convert_element_type_380 = None
        unsqueeze_316: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 4);  unsqueeze_315 = None
        view_468: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_316, [1, 16384, 1024]);  unsqueeze_316 = None
        squeeze_dim_112: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_468, 0);  view_468 = None
        unsqueeze_317: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg185_1, 3);  arg185_1 = None
        unsqueeze_318: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 4);  unsqueeze_317 = None
        view_469: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_318, [1, 1024, 1024]);  unsqueeze_318 = None
        squeeze_dim_113: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_469, 0);  view_469 = None
        mm_default_56: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_112, squeeze_dim_113);  squeeze_dim_112 = squeeze_dim_113 = None
        unsqueeze_default_56: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_56, 0);  mm_default_56 = None
        view_470: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_56, [1024, 16, 1, 16, 64]);  unsqueeze_default_56 = None
        permute_526: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_470, [0, 1, 3, 4, 2]);  view_470 = None
        view_471: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_526, [1024, 16, 16, 64]);  permute_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_322: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_471, 4);  view_471 = None
        permute_533: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_322, [1, 2, 4, 0, 3]);  unsqueeze_322 = None
        permute_535: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_533, [0, 1, 4, 3, 2]);  permute_533 = None
        view_477: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_535, [256, 64, 1024]);  permute_535 = None
        bmm_101: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_476, view_477);  view_476 = view_477 = None
        view_478: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_101, [16, 16, 512, 1, 1024]);  bmm_101 = None
        permute_536: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_478, [0, 1, 2, 4, 3]);  view_478 = None
        view_479: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_536, [16, 16, 512, 1024]);  permute_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_480: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_479, [16, 16, 1024, 512]);  view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_26: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_480, 2, 1, 9223372036854775807);  view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_481: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_26, [16, 16, 512, 1023]);  slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_14: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_12: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_481, [None, None, None, iota_14]);  view_481 = iota_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_134: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_475, index_12);  view_475 = index_12 = None
        add_135: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, 0);  add_134 = None

        # No stacktrace found for following nodes
        mul_tensor_44: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_135, 0.125)
        convert_element_type_default_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_44, torch.float32);  mul_tensor_44 = None
        eq_tensor_11: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_22, convert_element_type_default_22)
        abs_default_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_22)
        ne_scalar_11: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_11, inf);  abs_default_11 = None
        mul_tensor_47: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_11, ne_scalar_11);  eq_tensor_11 = ne_scalar_11 = None
        logical_not_default_22: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_47);  mul_tensor_47 = None
        any_dims_11: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_22, [3], True);  logical_not_default_22 = None
        logical_not_default_23: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_11);  any_dims_11 = None
        convert_element_type_default_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.float32);  add_135 = None
        mul_tensor_45: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_23, 1);  convert_element_type_default_23 = None
        amax_default_22: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_45, [3], True)
        sub_tensor_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_45, amax_default_22);  mul_tensor_45 = amax_default_22 = None
        mul_tensor_46: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_22, 0.125);  sub_tensor_22 = None
        amax_default_23: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_22, [3], True)
        sub_tensor_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_22, amax_default_23);  convert_element_type_default_22 = amax_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_23, mul_tensor_46, sub_tensor_23);  logical_not_default_23 = mul_tensor_46 = sub_tensor_23 = None
        exp_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_11);  where_self_11 = None
        sum_13: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [3], True)
        div_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_388: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_323: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_388, 4);  convert_element_type_388 = None
        view_482: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_323, [256, 512, 512]);  unsqueeze_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_311: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_373, 3)
        unsqueeze_312: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 4);  unsqueeze_311 = None
        view_464: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_312, [1, 8192, 1024]);  unsqueeze_312 = None
        squeeze_dim_114: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_464, 0);  view_464 = None
        unsqueeze_313: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg184_1, 3);  arg184_1 = None
        unsqueeze_314: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 4);  unsqueeze_313 = None
        view_465: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_314, [1, 1024, 1024]);  unsqueeze_314 = None
        squeeze_dim_115: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_465, 0);  view_465 = None
        mm_default_57: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_114, squeeze_dim_115);  squeeze_dim_114 = squeeze_dim_115 = None
        unsqueeze_default_57: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_57, 0);  mm_default_57 = None
        view_466: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_57, [512, 16, 1, 16, 64]);  unsqueeze_default_57 = None
        permute_521: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 1, 3, 4, 2]);  view_466 = None
        view_467: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_521, [512, 16, 16, 64]);  permute_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_324: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_467, 4);  view_467 = None
        permute_538: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_324, [4, 1, 2, 3, 0]);  unsqueeze_324 = None
        permute_540: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_538, [1, 2, 4, 3, 0]);  permute_538 = None
        view_483: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_540, [256, 512, 64]);  permute_540 = None
        bmm_102: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_482, view_483);  view_482 = view_483 = None
        view_484: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_102, [16, 16, 512, 1, 64]);  bmm_102 = None
        permute_541: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_484, [2, 0, 1, 4, 3]);  view_484 = None
        view_485: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_541, [512, 16, 16, 64]);  permute_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_325: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_485, 4);  view_485 = None
        permute_542: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_325, [0, 1, 4, 3, 2]);  unsqueeze_325 = None
        permute_544: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_542, [0, 1, 3, 4, 2]);  permute_542 = None
        clone_76: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_544, memory_format = torch.contiguous_format);  permute_544 = None
        view_486: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [1, 8192, 1024]);  clone_76 = None
        squeeze_dim_110: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_486, 0);  view_486 = None
        unsqueeze_326: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg188_1, 3);  arg188_1 = None
        unsqueeze_327: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 4);  unsqueeze_326 = None
        permute_543: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_327, [3, 4, 0, 2, 1]);  unsqueeze_327 = None
        permute_545: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_543, [3, 4, 2, 0, 1]);  permute_543 = None
        clone_77: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_545, memory_format = torch.contiguous_format);  permute_545 = None
        view_487: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [1, 1024, 1024]);  clone_77 = None
        squeeze_dim_111: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_487, 0);  view_487 = None
        mm_default_55: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_110, squeeze_dim_111);  squeeze_dim_110 = squeeze_dim_111 = None
        unsqueeze_default_55: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_55, 0);  mm_default_55 = None
        view_488: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_55, [512, 16, 1, 1, 1024]);  unsqueeze_default_55 = None
        permute_546: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_488, [0, 1, 4, 2, 3]);  view_488 = None
        view_489: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_546, [512, 16, 1024]);  permute_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_136: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_489, convert_element_type_373);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_393: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.float32);  add_136 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_393, [2], correction = 0, keepdim = True)
        getitem_48: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        sub_37: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_393, getitem_49);  convert_element_type_393 = getitem_49 = None
        add_137: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        mul_99: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
        mul_100: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, arg189_1);  mul_99 = arg189_1 = None
        add_138: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_100, arg190_1);  mul_100 = arg190_1 = None
        convert_element_type_394: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16);  add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_490: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_394, [8192, 1024])
        permute_547: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_24: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, view_490, permute_547);  arg192_1 = view_490 = permute_547 = None
        view_491: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [512, 16, 4096]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_398: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_491, torch.float32);  view_491 = None
        mul_101: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_398, 0.5)
        mul_102: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_398, 0.7071067811865476);  convert_element_type_398 = None
        erf_12: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_102);  mul_102 = None
        add_139: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_103: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, add_139);  mul_101 = add_139 = None
        convert_element_type_399: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_103, torch.bfloat16);  mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_492: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_399, [8192, 4096]);  convert_element_type_399 = None
        permute_548: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_25: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg194_1, view_492, permute_548);  arg194_1 = view_492 = permute_548 = None
        view_493: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [512, 16, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_140: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_493, convert_element_type_394);  view_493 = convert_element_type_394 = None
        convert_element_type_403: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_140, torch.float32);  add_140 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_403, [2], correction = 0, keepdim = True)
        getitem_50: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        sub_38: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_403, getitem_51);  convert_element_type_403 = getitem_51 = None
        add_141: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_104: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
        mul_105: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, arg195_1);  mul_104 = arg195_1 = None
        add_142: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_105, arg196_1);  mul_105 = arg196_1 = None
        convert_element_type_404: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_328: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_404, 3)
        unsqueeze_329: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 4);  unsqueeze_328 = None
        view_494: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_329, [1, 8192, 1024]);  unsqueeze_329 = None
        squeeze_dim_108: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_494, 0);  view_494 = None
        unsqueeze_330: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg197_1, 3);  arg197_1 = None
        unsqueeze_331: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 4);  unsqueeze_330 = None
        view_495: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_331, [1, 1024, 1024]);  unsqueeze_331 = None
        squeeze_dim_109: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_495, 0);  view_495 = None
        mm_default_54: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_108, squeeze_dim_109);  squeeze_dim_108 = squeeze_dim_109 = None
        unsqueeze_default_54: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_54, 0);  mm_default_54 = None
        view_496: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_54, [512, 16, 1, 16, 64]);  unsqueeze_default_54 = None
        permute_553: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_496, [0, 1, 3, 4, 2]);  view_496 = None
        view_497: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_553, [512, 16, 16, 64]);  permute_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_143: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_497, arg201_1);  arg201_1 = None
        unsqueeze_344: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_143, 4);  add_143 = None
        permute_569: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_344, [1, 2, 0, 4, 3]);  unsqueeze_344 = None
        permute_571: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_569, [0, 1, 2, 4, 3]);  permute_569 = None
        view_510: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_571, [256, 512, 64]);  permute_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_332: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_404, 3)
        unsqueeze_333: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 4);  unsqueeze_332 = None
        view_498: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_333, [1, 8192, 1024]);  unsqueeze_333 = None
        squeeze_dim_106: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_498, 0);  view_498 = None
        unsqueeze_334: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg198_1, 3);  arg198_1 = None
        unsqueeze_335: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 4);  unsqueeze_334 = None
        view_499: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_335, [1, 1024, 1024]);  unsqueeze_335 = None
        squeeze_dim_107: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_499, 0);  view_499 = None
        mm_default_53: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_106, squeeze_dim_107);  squeeze_dim_106 = squeeze_dim_107 = None
        unsqueeze_default_53: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_53, 0);  mm_default_53 = None
        view_500: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_53, [512, 16, 1, 16, 64]);  unsqueeze_default_53 = None
        permute_558: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_500, [0, 1, 3, 4, 2]);  view_500 = None
        view_501: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_558, [512, 16, 16, 64]);  permute_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_345: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_501, 4);  view_501 = None
        permute_570: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_345, [1, 2, 4, 0, 3]);  unsqueeze_345 = None
        permute_572: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_570, [0, 1, 4, 3, 2]);  permute_570 = None
        view_511: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_572, [256, 64, 512]);  permute_572 = None
        bmm_108: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_510, view_511);  view_510 = view_511 = None
        view_512: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_108, [16, 16, 512, 1, 512]);  bmm_108 = None
        permute_573: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 1, 2, 4, 3]);  view_512 = None
        view_513: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_573, [16, 16, 512, 512]);  permute_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_144: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_497, arg202_1);  view_497 = arg202_1 = None
        unsqueeze_346: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_144, 4);  add_144 = None
        permute_574: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_346, [1, 2, 0, 4, 3]);  unsqueeze_346 = None
        permute_576: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_574, [0, 1, 2, 4, 3]);  permute_574 = None
        view_514: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_576, [256, 512, 64]);  permute_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_411: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_340: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_411, 3);  convert_element_type_411 = None
        unsqueeze_341: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 4);  unsqueeze_340 = None
        view_506: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_341, [1, 16384, 1024]);  unsqueeze_341 = None
        squeeze_dim_102: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_506, 0);  view_506 = None
        unsqueeze_342: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg200_1, 3);  arg200_1 = None
        unsqueeze_343: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 4);  unsqueeze_342 = None
        view_507: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_343, [1, 1024, 1024]);  unsqueeze_343 = None
        squeeze_dim_103: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_507, 0);  view_507 = None
        mm_default_51: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_102, squeeze_dim_103);  squeeze_dim_102 = squeeze_dim_103 = None
        unsqueeze_default_51: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_51, 0);  mm_default_51 = None
        view_508: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_51, [1024, 16, 1, 16, 64]);  unsqueeze_default_51 = None
        permute_568: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_508, [0, 1, 3, 4, 2]);  view_508 = None
        view_509: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_568, [1024, 16, 16, 64]);  permute_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_347: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_509, 4);  view_509 = None
        permute_575: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_347, [1, 2, 4, 0, 3]);  unsqueeze_347 = None
        permute_577: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_575, [0, 1, 4, 3, 2]);  permute_575 = None
        view_515: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_577, [256, 64, 1024]);  permute_577 = None
        bmm_109: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_514, view_515);  view_514 = view_515 = None
        view_516: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_109, [16, 16, 512, 1, 1024]);  bmm_109 = None
        permute_578: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_516, [0, 1, 2, 4, 3]);  view_516 = None
        view_517: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_578, [16, 16, 512, 1024]);  permute_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_518: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_517, [16, 16, 1024, 512]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_28: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_518, 2, 1, 9223372036854775807);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_519: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_28, [16, 16, 512, 1023]);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_15: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_13: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_519, [None, None, None, iota_15]);  view_519 = iota_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_145: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_513, index_13);  view_513 = index_13 = None
        add_146: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, 0);  add_145 = None

        # No stacktrace found for following nodes
        mul_tensor_40: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_146, 0.125)
        convert_element_type_default_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_40, torch.float32);  mul_tensor_40 = None
        eq_tensor_10: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_20, convert_element_type_default_20)
        abs_default_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_20)
        ne_scalar_10: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_10, inf);  abs_default_10 = None
        mul_tensor_43: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_10, ne_scalar_10);  eq_tensor_10 = ne_scalar_10 = None
        logical_not_default_20: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_43);  mul_tensor_43 = None
        any_dims_10: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_20, [3], True);  logical_not_default_20 = None
        logical_not_default_21: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_10);  any_dims_10 = None
        convert_element_type_default_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_146, torch.float32);  add_146 = None
        mul_tensor_41: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_21, 1);  convert_element_type_default_21 = None
        amax_default_20: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_41, [3], True)
        sub_tensor_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_41, amax_default_20);  mul_tensor_41 = amax_default_20 = None
        mul_tensor_42: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_20, 0.125);  sub_tensor_20 = None
        amax_default_21: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_20, [3], True)
        sub_tensor_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_20, amax_default_21);  convert_element_type_default_20 = amax_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_21, mul_tensor_42, sub_tensor_21);  logical_not_default_21 = mul_tensor_42 = sub_tensor_21 = None
        exp_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_10);  where_self_10 = None
        sum_14: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [3], True)
        div_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_419: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_348: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_419, 4);  convert_element_type_419 = None
        view_520: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_348, [256, 512, 512]);  unsqueeze_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_336: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_404, 3)
        unsqueeze_337: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 4);  unsqueeze_336 = None
        view_502: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_337, [1, 8192, 1024]);  unsqueeze_337 = None
        squeeze_dim_104: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_502, 0);  view_502 = None
        unsqueeze_338: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg199_1, 3);  arg199_1 = None
        unsqueeze_339: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 4);  unsqueeze_338 = None
        view_503: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_339, [1, 1024, 1024]);  unsqueeze_339 = None
        squeeze_dim_105: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_503, 0);  view_503 = None
        mm_default_52: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_104, squeeze_dim_105);  squeeze_dim_104 = squeeze_dim_105 = None
        unsqueeze_default_52: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_52, 0);  mm_default_52 = None
        view_504: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_52, [512, 16, 1, 16, 64]);  unsqueeze_default_52 = None
        permute_563: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_504, [0, 1, 3, 4, 2]);  view_504 = None
        view_505: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_563, [512, 16, 16, 64]);  permute_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_349: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_505, 4);  view_505 = None
        permute_580: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_349, [4, 1, 2, 3, 0]);  unsqueeze_349 = None
        permute_582: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_580, [1, 2, 4, 3, 0]);  permute_580 = None
        view_521: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_582, [256, 512, 64]);  permute_582 = None
        bmm_110: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_520, view_521);  view_520 = view_521 = None
        view_522: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_110, [16, 16, 512, 1, 64]);  bmm_110 = None
        permute_583: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_522, [2, 0, 1, 4, 3]);  view_522 = None
        view_523: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_583, [512, 16, 16, 64]);  permute_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_350: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_523, 4);  view_523 = None
        permute_584: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_350, [0, 1, 4, 3, 2]);  unsqueeze_350 = None
        permute_586: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_584, [0, 1, 3, 4, 2]);  permute_584 = None
        clone_82: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
        view_524: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [1, 8192, 1024]);  clone_82 = None
        squeeze_dim_100: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_524, 0);  view_524 = None
        unsqueeze_351: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg203_1, 3);  arg203_1 = None
        unsqueeze_352: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 4);  unsqueeze_351 = None
        permute_585: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_352, [3, 4, 0, 2, 1]);  unsqueeze_352 = None
        permute_587: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_585, [3, 4, 2, 0, 1]);  permute_585 = None
        clone_83: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
        view_525: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [1, 1024, 1024]);  clone_83 = None
        squeeze_dim_101: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_525, 0);  view_525 = None
        mm_default_50: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_100, squeeze_dim_101);  squeeze_dim_100 = squeeze_dim_101 = None
        unsqueeze_default_50: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_50, 0);  mm_default_50 = None
        view_526: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_50, [512, 16, 1, 1, 1024]);  unsqueeze_default_50 = None
        permute_588: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_526, [0, 1, 4, 2, 3]);  view_526 = None
        view_527: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_588, [512, 16, 1024]);  permute_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_147: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_527, convert_element_type_404);  view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_424: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.float32);  add_147 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_424, [2], correction = 0, keepdim = True)
        getitem_52: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        sub_40: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_424, getitem_53);  convert_element_type_424 = getitem_53 = None
        add_148: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_107: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = rsqrt_26 = None
        mul_108: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, arg204_1);  mul_107 = arg204_1 = None
        add_149: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_108, arg205_1);  mul_108 = arg205_1 = None
        convert_element_type_425: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_528: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_425, [8192, 1024])
        permute_589: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_26: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg207_1, view_528, permute_589);  arg207_1 = view_528 = permute_589 = None
        view_529: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [512, 16, 4096]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_429: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        mul_109: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_429, 0.5)
        mul_110: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_429, 0.7071067811865476);  convert_element_type_429 = None
        erf_13: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
        add_150: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_111: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, add_150);  mul_109 = add_150 = None
        convert_element_type_430: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_111, torch.bfloat16);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_530: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_430, [8192, 4096]);  convert_element_type_430 = None
        permute_590: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_27: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg209_1, view_530, permute_590);  arg209_1 = view_530 = permute_590 = None
        view_531: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [512, 16, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_151: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_531, convert_element_type_425);  view_531 = convert_element_type_425 = None
        convert_element_type_434: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.float32);  add_151 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_434, [2], correction = 0, keepdim = True)
        getitem_54: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        sub_41: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_434, getitem_55);  convert_element_type_434 = getitem_55 = None
        add_152: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_112: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_113: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, arg210_1);  mul_112 = arg210_1 = None
        add_153: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, arg211_1);  mul_113 = arg211_1 = None
        convert_element_type_435: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_353: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_435, 3)
        unsqueeze_354: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 4);  unsqueeze_353 = None
        view_532: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_354, [1, 8192, 1024]);  unsqueeze_354 = None
        squeeze_dim_98: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_532, 0);  view_532 = None
        unsqueeze_355: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg212_1, 3);  arg212_1 = None
        unsqueeze_356: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 4);  unsqueeze_355 = None
        view_533: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_356, [1, 1024, 1024]);  unsqueeze_356 = None
        squeeze_dim_99: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_533, 0);  view_533 = None
        mm_default_49: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_98, squeeze_dim_99);  squeeze_dim_98 = squeeze_dim_99 = None
        unsqueeze_default_49: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_49, 0);  mm_default_49 = None
        view_534: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_49, [512, 16, 1, 16, 64]);  unsqueeze_default_49 = None
        permute_595: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_534, [0, 1, 3, 4, 2]);  view_534 = None
        view_535: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_595, [512, 16, 16, 64]);  permute_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_154: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_535, arg216_1);  arg216_1 = None
        unsqueeze_369: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_154, 4);  add_154 = None
        permute_611: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_369, [1, 2, 0, 4, 3]);  unsqueeze_369 = None
        permute_613: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_611, [0, 1, 2, 4, 3]);  permute_611 = None
        view_548: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_613, [256, 512, 64]);  permute_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_357: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_435, 3)
        unsqueeze_358: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 4);  unsqueeze_357 = None
        view_536: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_358, [1, 8192, 1024]);  unsqueeze_358 = None
        squeeze_dim_96: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_536, 0);  view_536 = None
        unsqueeze_359: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg213_1, 3);  arg213_1 = None
        unsqueeze_360: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 4);  unsqueeze_359 = None
        view_537: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_360, [1, 1024, 1024]);  unsqueeze_360 = None
        squeeze_dim_97: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_537, 0);  view_537 = None
        mm_default_48: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_96, squeeze_dim_97);  squeeze_dim_96 = squeeze_dim_97 = None
        unsqueeze_default_48: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_48, 0);  mm_default_48 = None
        view_538: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_48, [512, 16, 1, 16, 64]);  unsqueeze_default_48 = None
        permute_600: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_538, [0, 1, 3, 4, 2]);  view_538 = None
        view_539: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_600, [512, 16, 16, 64]);  permute_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_370: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_539, 4);  view_539 = None
        permute_612: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_370, [1, 2, 4, 0, 3]);  unsqueeze_370 = None
        permute_614: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_612, [0, 1, 4, 3, 2]);  permute_612 = None
        view_549: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_614, [256, 64, 512]);  permute_614 = None
        bmm_116: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_548, view_549);  view_548 = view_549 = None
        view_550: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_116, [16, 16, 512, 1, 512]);  bmm_116 = None
        permute_615: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_550, [0, 1, 2, 4, 3]);  view_550 = None
        view_551: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_615, [16, 16, 512, 512]);  permute_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_155: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_535, arg217_1);  view_535 = arg217_1 = None
        unsqueeze_371: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_155, 4);  add_155 = None
        permute_616: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_371, [1, 2, 0, 4, 3]);  unsqueeze_371 = None
        permute_618: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_616, [0, 1, 2, 4, 3]);  permute_616 = None
        view_552: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_618, [256, 512, 64]);  permute_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_442: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_365: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_442, 3);  convert_element_type_442 = None
        unsqueeze_366: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 4);  unsqueeze_365 = None
        view_544: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_366, [1, 16384, 1024]);  unsqueeze_366 = None
        squeeze_dim_92: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_544, 0);  view_544 = None
        unsqueeze_367: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg215_1, 3);  arg215_1 = None
        unsqueeze_368: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 4);  unsqueeze_367 = None
        view_545: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_368, [1, 1024, 1024]);  unsqueeze_368 = None
        squeeze_dim_93: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_545, 0);  view_545 = None
        mm_default_46: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_92, squeeze_dim_93);  squeeze_dim_92 = squeeze_dim_93 = None
        unsqueeze_default_46: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_46, 0);  mm_default_46 = None
        view_546: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_46, [1024, 16, 1, 16, 64]);  unsqueeze_default_46 = None
        permute_610: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_546, [0, 1, 3, 4, 2]);  view_546 = None
        view_547: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_610, [1024, 16, 16, 64]);  permute_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_372: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_547, 4);  view_547 = None
        permute_617: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_372, [1, 2, 4, 0, 3]);  unsqueeze_372 = None
        permute_619: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_617, [0, 1, 4, 3, 2]);  permute_617 = None
        view_553: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_619, [256, 64, 1024]);  permute_619 = None
        bmm_117: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_552, view_553);  view_552 = view_553 = None
        view_554: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_117, [16, 16, 512, 1, 1024]);  bmm_117 = None
        permute_620: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_554, [0, 1, 2, 4, 3]);  view_554 = None
        view_555: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_620, [16, 16, 512, 1024]);  permute_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_556: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_555, [16, 16, 1024, 512]);  view_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_30: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_556, 2, 1, 9223372036854775807);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_557: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_30, [16, 16, 512, 1023]);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_16: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_14: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_557, [None, None, None, iota_16]);  view_557 = iota_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_156: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_551, index_14);  view_551 = index_14 = None
        add_157: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_156, 0);  add_156 = None

        # No stacktrace found for following nodes
        mul_tensor_36: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_157, 0.125)
        convert_element_type_default_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_36, torch.float32);  mul_tensor_36 = None
        eq_tensor_9: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_18, convert_element_type_default_18)
        abs_default_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_18)
        ne_scalar_9: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_9, inf);  abs_default_9 = None
        mul_tensor_39: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_9, ne_scalar_9);  eq_tensor_9 = ne_scalar_9 = None
        logical_not_default_18: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_39);  mul_tensor_39 = None
        any_dims_9: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_18, [3], True);  logical_not_default_18 = None
        logical_not_default_19: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_9);  any_dims_9 = None
        convert_element_type_default_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.float32);  add_157 = None
        mul_tensor_37: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_19, 1);  convert_element_type_default_19 = None
        amax_default_18: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_37, [3], True)
        sub_tensor_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_37, amax_default_18);  mul_tensor_37 = amax_default_18 = None
        mul_tensor_38: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_18, 0.125);  sub_tensor_18 = None
        amax_default_19: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_18, [3], True)
        sub_tensor_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_18, amax_default_19);  convert_element_type_default_18 = amax_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_19, mul_tensor_38, sub_tensor_19);  logical_not_default_19 = mul_tensor_38 = sub_tensor_19 = None
        exp_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_9);  where_self_9 = None
        sum_15: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [3], True)
        div_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_450: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_373: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_450, 4);  convert_element_type_450 = None
        view_558: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_373, [256, 512, 512]);  unsqueeze_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_361: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_435, 3)
        unsqueeze_362: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 4);  unsqueeze_361 = None
        view_540: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_362, [1, 8192, 1024]);  unsqueeze_362 = None
        squeeze_dim_94: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_540, 0);  view_540 = None
        unsqueeze_363: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg214_1, 3);  arg214_1 = None
        unsqueeze_364: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 4);  unsqueeze_363 = None
        view_541: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_364, [1, 1024, 1024]);  unsqueeze_364 = None
        squeeze_dim_95: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_541, 0);  view_541 = None
        mm_default_47: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_94, squeeze_dim_95);  squeeze_dim_94 = squeeze_dim_95 = None
        unsqueeze_default_47: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_47, 0);  mm_default_47 = None
        view_542: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_47, [512, 16, 1, 16, 64]);  unsqueeze_default_47 = None
        permute_605: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_542, [0, 1, 3, 4, 2]);  view_542 = None
        view_543: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_605, [512, 16, 16, 64]);  permute_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_374: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_543, 4);  view_543 = None
        permute_622: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_374, [4, 1, 2, 3, 0]);  unsqueeze_374 = None
        permute_624: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_622, [1, 2, 4, 3, 0]);  permute_622 = None
        view_559: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_624, [256, 512, 64]);  permute_624 = None
        bmm_118: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_558, view_559);  view_558 = view_559 = None
        view_560: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_118, [16, 16, 512, 1, 64]);  bmm_118 = None
        permute_625: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_560, [2, 0, 1, 4, 3]);  view_560 = None
        view_561: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_625, [512, 16, 16, 64]);  permute_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_375: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_561, 4);  view_561 = None
        permute_626: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_375, [0, 1, 4, 3, 2]);  unsqueeze_375 = None
        permute_628: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_626, [0, 1, 3, 4, 2]);  permute_626 = None
        clone_88: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_628, memory_format = torch.contiguous_format);  permute_628 = None
        view_562: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [1, 8192, 1024]);  clone_88 = None
        squeeze_dim_90: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_562, 0);  view_562 = None
        unsqueeze_376: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg218_1, 3);  arg218_1 = None
        unsqueeze_377: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 4);  unsqueeze_376 = None
        permute_627: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_377, [3, 4, 0, 2, 1]);  unsqueeze_377 = None
        permute_629: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_627, [3, 4, 2, 0, 1]);  permute_627 = None
        clone_89: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
        view_563: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [1, 1024, 1024]);  clone_89 = None
        squeeze_dim_91: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_563, 0);  view_563 = None
        mm_default_45: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_90, squeeze_dim_91);  squeeze_dim_90 = squeeze_dim_91 = None
        unsqueeze_default_45: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_45, 0);  mm_default_45 = None
        view_564: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_45, [512, 16, 1, 1, 1024]);  unsqueeze_default_45 = None
        permute_630: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_564, [0, 1, 4, 2, 3]);  view_564 = None
        view_565: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_630, [512, 16, 1024]);  permute_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_158: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_565, convert_element_type_435);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_455: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.float32);  add_158 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_455, [2], correction = 0, keepdim = True)
        getitem_56: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        sub_43: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_455, getitem_57);  convert_element_type_455 = getitem_57 = None
        add_159: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_115: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = rsqrt_28 = None
        mul_116: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, arg219_1);  mul_115 = arg219_1 = None
        add_160: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, arg220_1);  mul_116 = arg220_1 = None
        convert_element_type_456: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_566: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_456, [8192, 1024])
        permute_631: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_28: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg222_1, view_566, permute_631);  arg222_1 = view_566 = permute_631 = None
        view_567: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [512, 16, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_460: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_567, torch.float32);  view_567 = None
        mul_117: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_460, 0.5)
        mul_118: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_460, 0.7071067811865476);  convert_element_type_460 = None
        erf_14: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_161: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_119: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, add_161);  mul_117 = add_161 = None
        convert_element_type_461: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_568: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_461, [8192, 4096]);  convert_element_type_461 = None
        permute_632: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_29: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg224_1, view_568, permute_632);  arg224_1 = view_568 = permute_632 = None
        view_569: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [512, 16, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_162: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_569, convert_element_type_456);  view_569 = convert_element_type_456 = None
        convert_element_type_465: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_162, torch.float32);  add_162 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_465, [2], correction = 0, keepdim = True)
        getitem_58: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        sub_44: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_465, getitem_59);  convert_element_type_465 = getitem_59 = None
        add_163: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_120: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_121: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, arg225_1);  mul_120 = arg225_1 = None
        add_164: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, arg226_1);  mul_121 = arg226_1 = None
        convert_element_type_466: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_378: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_466, 3)
        unsqueeze_379: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 4);  unsqueeze_378 = None
        view_570: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_379, [1, 8192, 1024]);  unsqueeze_379 = None
        squeeze_dim_88: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_570, 0);  view_570 = None
        unsqueeze_380: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg227_1, 3);  arg227_1 = None
        unsqueeze_381: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 4);  unsqueeze_380 = None
        view_571: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_381, [1, 1024, 1024]);  unsqueeze_381 = None
        squeeze_dim_89: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_571, 0);  view_571 = None
        mm_default_44: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_88, squeeze_dim_89);  squeeze_dim_88 = squeeze_dim_89 = None
        unsqueeze_default_44: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_44, 0);  mm_default_44 = None
        view_572: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_44, [512, 16, 1, 16, 64]);  unsqueeze_default_44 = None
        permute_637: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_572, [0, 1, 3, 4, 2]);  view_572 = None
        view_573: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_637, [512, 16, 16, 64]);  permute_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_165: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_573, arg231_1);  arg231_1 = None
        unsqueeze_394: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_165, 4);  add_165 = None
        permute_653: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_394, [1, 2, 0, 4, 3]);  unsqueeze_394 = None
        permute_655: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_653, [0, 1, 2, 4, 3]);  permute_653 = None
        view_586: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_655, [256, 512, 64]);  permute_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_382: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_466, 3)
        unsqueeze_383: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 4);  unsqueeze_382 = None
        view_574: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_383, [1, 8192, 1024]);  unsqueeze_383 = None
        squeeze_dim_86: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_574, 0);  view_574 = None
        unsqueeze_384: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg228_1, 3);  arg228_1 = None
        unsqueeze_385: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 4);  unsqueeze_384 = None
        view_575: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_385, [1, 1024, 1024]);  unsqueeze_385 = None
        squeeze_dim_87: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_575, 0);  view_575 = None
        mm_default_43: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_86, squeeze_dim_87);  squeeze_dim_86 = squeeze_dim_87 = None
        unsqueeze_default_43: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_43, 0);  mm_default_43 = None
        view_576: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_43, [512, 16, 1, 16, 64]);  unsqueeze_default_43 = None
        permute_642: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_576, [0, 1, 3, 4, 2]);  view_576 = None
        view_577: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_642, [512, 16, 16, 64]);  permute_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_395: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_577, 4);  view_577 = None
        permute_654: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_395, [1, 2, 4, 0, 3]);  unsqueeze_395 = None
        permute_656: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_654, [0, 1, 4, 3, 2]);  permute_654 = None
        view_587: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_656, [256, 64, 512]);  permute_656 = None
        bmm_124: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_586, view_587);  view_586 = view_587 = None
        view_588: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_124, [16, 16, 512, 1, 512]);  bmm_124 = None
        permute_657: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_588, [0, 1, 2, 4, 3]);  view_588 = None
        view_589: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_657, [16, 16, 512, 512]);  permute_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_166: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_573, arg232_1);  view_573 = arg232_1 = None
        unsqueeze_396: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_166, 4);  add_166 = None
        permute_658: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_396, [1, 2, 0, 4, 3]);  unsqueeze_396 = None
        permute_660: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_658, [0, 1, 2, 4, 3]);  permute_658 = None
        view_590: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_660, [256, 512, 64]);  permute_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_473: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_390: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_473, 3);  convert_element_type_473 = None
        unsqueeze_391: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 4);  unsqueeze_390 = None
        view_582: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_391, [1, 16384, 1024]);  unsqueeze_391 = None
        squeeze_dim_82: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_582, 0);  view_582 = None
        unsqueeze_392: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg230_1, 3);  arg230_1 = None
        unsqueeze_393: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 4);  unsqueeze_392 = None
        view_583: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_393, [1, 1024, 1024]);  unsqueeze_393 = None
        squeeze_dim_83: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_583, 0);  view_583 = None
        mm_default_41: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_82, squeeze_dim_83);  squeeze_dim_82 = squeeze_dim_83 = None
        unsqueeze_default_41: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_41, 0);  mm_default_41 = None
        view_584: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_41, [1024, 16, 1, 16, 64]);  unsqueeze_default_41 = None
        permute_652: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_584, [0, 1, 3, 4, 2]);  view_584 = None
        view_585: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_652, [1024, 16, 16, 64]);  permute_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_397: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_585, 4);  view_585 = None
        permute_659: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_397, [1, 2, 4, 0, 3]);  unsqueeze_397 = None
        permute_661: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_659, [0, 1, 4, 3, 2]);  permute_659 = None
        view_591: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_661, [256, 64, 1024]);  permute_661 = None
        bmm_125: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_590, view_591);  view_590 = view_591 = None
        view_592: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_125, [16, 16, 512, 1, 1024]);  bmm_125 = None
        permute_662: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_592, [0, 1, 2, 4, 3]);  view_592 = None
        view_593: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_662, [16, 16, 512, 1024]);  permute_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_594: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [16, 16, 1024, 512]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_32: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_594, 2, 1, 9223372036854775807);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_595: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_32, [16, 16, 512, 1023]);  slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_17: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_15: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_595, [None, None, None, iota_17]);  view_595 = iota_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_167: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_589, index_15);  view_589 = index_15 = None
        add_168: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_167, 0);  add_167 = None

        # No stacktrace found for following nodes
        mul_tensor_32: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, 0.125)
        convert_element_type_default_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_32, torch.float32);  mul_tensor_32 = None
        eq_tensor_8: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_16, convert_element_type_default_16)
        abs_default_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_16)
        ne_scalar_8: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_8, inf);  abs_default_8 = None
        mul_tensor_35: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_8, ne_scalar_8);  eq_tensor_8 = ne_scalar_8 = None
        logical_not_default_16: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_35);  mul_tensor_35 = None
        any_dims_8: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_16, [3], True);  logical_not_default_16 = None
        logical_not_default_17: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_8);  any_dims_8 = None
        convert_element_type_default_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_168, torch.float32);  add_168 = None
        mul_tensor_33: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_17, 1);  convert_element_type_default_17 = None
        amax_default_16: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_33, [3], True)
        sub_tensor_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_33, amax_default_16);  mul_tensor_33 = amax_default_16 = None
        mul_tensor_34: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_16, 0.125);  sub_tensor_16 = None
        amax_default_17: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_16, [3], True)
        sub_tensor_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_16, amax_default_17);  convert_element_type_default_16 = amax_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_17, mul_tensor_34, sub_tensor_17);  logical_not_default_17 = mul_tensor_34 = sub_tensor_17 = None
        exp_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_8);  where_self_8 = None
        sum_16: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [3], True)
        div_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_481: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_398: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_481, 4);  convert_element_type_481 = None
        view_596: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_398, [256, 512, 512]);  unsqueeze_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_386: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_466, 3)
        unsqueeze_387: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 4);  unsqueeze_386 = None
        view_578: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_387, [1, 8192, 1024]);  unsqueeze_387 = None
        squeeze_dim_84: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_578, 0);  view_578 = None
        unsqueeze_388: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg229_1, 3);  arg229_1 = None
        unsqueeze_389: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 4);  unsqueeze_388 = None
        view_579: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_389, [1, 1024, 1024]);  unsqueeze_389 = None
        squeeze_dim_85: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_579, 0);  view_579 = None
        mm_default_42: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_84, squeeze_dim_85);  squeeze_dim_84 = squeeze_dim_85 = None
        unsqueeze_default_42: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_42, 0);  mm_default_42 = None
        view_580: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_42, [512, 16, 1, 16, 64]);  unsqueeze_default_42 = None
        permute_647: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_580, [0, 1, 3, 4, 2]);  view_580 = None
        view_581: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_647, [512, 16, 16, 64]);  permute_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_399: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_581, 4);  view_581 = None
        permute_664: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_399, [4, 1, 2, 3, 0]);  unsqueeze_399 = None
        permute_666: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_664, [1, 2, 4, 3, 0]);  permute_664 = None
        view_597: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_666, [256, 512, 64]);  permute_666 = None
        bmm_126: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_596, view_597);  view_596 = view_597 = None
        view_598: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_126, [16, 16, 512, 1, 64]);  bmm_126 = None
        permute_667: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_598, [2, 0, 1, 4, 3]);  view_598 = None
        view_599: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_667, [512, 16, 16, 64]);  permute_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_400: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_599, 4);  view_599 = None
        permute_668: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_400, [0, 1, 4, 3, 2]);  unsqueeze_400 = None
        permute_670: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_668, [0, 1, 3, 4, 2]);  permute_668 = None
        clone_94: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_670, memory_format = torch.contiguous_format);  permute_670 = None
        view_600: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [1, 8192, 1024]);  clone_94 = None
        squeeze_dim_80: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_600, 0);  view_600 = None
        unsqueeze_401: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg233_1, 3);  arg233_1 = None
        unsqueeze_402: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 4);  unsqueeze_401 = None
        permute_669: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_402, [3, 4, 0, 2, 1]);  unsqueeze_402 = None
        permute_671: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_669, [3, 4, 2, 0, 1]);  permute_669 = None
        clone_95: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_671, memory_format = torch.contiguous_format);  permute_671 = None
        view_601: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [1, 1024, 1024]);  clone_95 = None
        squeeze_dim_81: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_601, 0);  view_601 = None
        mm_default_40: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_80, squeeze_dim_81);  squeeze_dim_80 = squeeze_dim_81 = None
        unsqueeze_default_40: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_40, 0);  mm_default_40 = None
        view_602: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_40, [512, 16, 1, 1, 1024]);  unsqueeze_default_40 = None
        permute_672: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_602, [0, 1, 4, 2, 3]);  view_602 = None
        view_603: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_672, [512, 16, 1024]);  permute_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_169: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_603, convert_element_type_466);  view_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_486: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.float32);  add_169 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_486, [2], correction = 0, keepdim = True)
        getitem_60: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        sub_46: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_486, getitem_61);  convert_element_type_486 = getitem_61 = None
        add_170: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_123: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = rsqrt_30 = None
        mul_124: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, arg234_1);  mul_123 = arg234_1 = None
        add_171: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_124, arg235_1);  mul_124 = arg235_1 = None
        convert_element_type_487: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16);  add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_604: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_487, [8192, 1024])
        permute_673: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_30: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg237_1, view_604, permute_673);  arg237_1 = view_604 = permute_673 = None
        view_605: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [512, 16, 4096]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_491: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_605, torch.float32);  view_605 = None
        mul_125: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_491, 0.5)
        mul_126: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_491, 0.7071067811865476);  convert_element_type_491 = None
        erf_15: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_172: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_127: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, add_172);  mul_125 = add_172 = None
        convert_element_type_492: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_127, torch.bfloat16);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_606: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_492, [8192, 4096]);  convert_element_type_492 = None
        permute_674: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_31: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg239_1, view_606, permute_674);  arg239_1 = view_606 = permute_674 = None
        view_607: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [512, 16, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_173: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_607, convert_element_type_487);  view_607 = convert_element_type_487 = None
        convert_element_type_496: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.float32);  add_173 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_496, [2], correction = 0, keepdim = True)
        getitem_62: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        sub_47: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_496, getitem_63);  convert_element_type_496 = getitem_63 = None
        add_174: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_128: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_129: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, arg240_1);  mul_128 = arg240_1 = None
        add_175: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_129, arg241_1);  mul_129 = arg241_1 = None
        convert_element_type_497: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_403: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_497, 3)
        unsqueeze_404: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 4);  unsqueeze_403 = None
        view_608: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_404, [1, 8192, 1024]);  unsqueeze_404 = None
        squeeze_dim_78: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_608, 0);  view_608 = None
        unsqueeze_405: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg242_1, 3);  arg242_1 = None
        unsqueeze_406: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 4);  unsqueeze_405 = None
        view_609: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_406, [1, 1024, 1024]);  unsqueeze_406 = None
        squeeze_dim_79: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_609, 0);  view_609 = None
        mm_default_39: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_78, squeeze_dim_79);  squeeze_dim_78 = squeeze_dim_79 = None
        unsqueeze_default_39: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_39, 0);  mm_default_39 = None
        view_610: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_39, [512, 16, 1, 16, 64]);  unsqueeze_default_39 = None
        permute_679: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_610, [0, 1, 3, 4, 2]);  view_610 = None
        view_611: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_679, [512, 16, 16, 64]);  permute_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_176: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_611, arg246_1);  arg246_1 = None
        unsqueeze_419: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_176, 4);  add_176 = None
        permute_695: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_419, [1, 2, 0, 4, 3]);  unsqueeze_419 = None
        permute_697: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_695, [0, 1, 2, 4, 3]);  permute_695 = None
        view_624: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_697, [256, 512, 64]);  permute_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_407: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_497, 3)
        unsqueeze_408: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 4);  unsqueeze_407 = None
        view_612: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_408, [1, 8192, 1024]);  unsqueeze_408 = None
        squeeze_dim_76: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_612, 0);  view_612 = None
        unsqueeze_409: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg243_1, 3);  arg243_1 = None
        unsqueeze_410: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 4);  unsqueeze_409 = None
        view_613: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_410, [1, 1024, 1024]);  unsqueeze_410 = None
        squeeze_dim_77: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_613, 0);  view_613 = None
        mm_default_38: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_76, squeeze_dim_77);  squeeze_dim_76 = squeeze_dim_77 = None
        unsqueeze_default_38: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_38, 0);  mm_default_38 = None
        view_614: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_38, [512, 16, 1, 16, 64]);  unsqueeze_default_38 = None
        permute_684: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_614, [0, 1, 3, 4, 2]);  view_614 = None
        view_615: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_684, [512, 16, 16, 64]);  permute_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_420: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_615, 4);  view_615 = None
        permute_696: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_420, [1, 2, 4, 0, 3]);  unsqueeze_420 = None
        permute_698: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_696, [0, 1, 4, 3, 2]);  permute_696 = None
        view_625: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_698, [256, 64, 512]);  permute_698 = None
        bmm_132: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_624, view_625);  view_624 = view_625 = None
        view_626: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_132, [16, 16, 512, 1, 512]);  bmm_132 = None
        permute_699: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_626, [0, 1, 2, 4, 3]);  view_626 = None
        view_627: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_699, [16, 16, 512, 512]);  permute_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_177: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_611, arg247_1);  view_611 = arg247_1 = None
        unsqueeze_421: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_177, 4);  add_177 = None
        permute_700: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_421, [1, 2, 0, 4, 3]);  unsqueeze_421 = None
        permute_702: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_700, [0, 1, 2, 4, 3]);  permute_700 = None
        view_628: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_702, [256, 512, 64]);  permute_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_504: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_415: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_504, 3);  convert_element_type_504 = None
        unsqueeze_416: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 4);  unsqueeze_415 = None
        view_620: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_416, [1, 16384, 1024]);  unsqueeze_416 = None
        squeeze_dim_72: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_620, 0);  view_620 = None
        unsqueeze_417: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg245_1, 3);  arg245_1 = None
        unsqueeze_418: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 4);  unsqueeze_417 = None
        view_621: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_418, [1, 1024, 1024]);  unsqueeze_418 = None
        squeeze_dim_73: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_621, 0);  view_621 = None
        mm_default_36: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_72, squeeze_dim_73);  squeeze_dim_72 = squeeze_dim_73 = None
        unsqueeze_default_36: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_36, 0);  mm_default_36 = None
        view_622: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_36, [1024, 16, 1, 16, 64]);  unsqueeze_default_36 = None
        permute_694: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_622, [0, 1, 3, 4, 2]);  view_622 = None
        view_623: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_694, [1024, 16, 16, 64]);  permute_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_422: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_623, 4);  view_623 = None
        permute_701: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_422, [1, 2, 4, 0, 3]);  unsqueeze_422 = None
        permute_703: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_701, [0, 1, 4, 3, 2]);  permute_701 = None
        view_629: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_703, [256, 64, 1024]);  permute_703 = None
        bmm_133: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_628, view_629);  view_628 = view_629 = None
        view_630: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_133, [16, 16, 512, 1, 1024]);  bmm_133 = None
        permute_704: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_630, [0, 1, 2, 4, 3]);  view_630 = None
        view_631: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_704, [16, 16, 512, 1024]);  permute_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_632: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_631, [16, 16, 1024, 512]);  view_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_34: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_632, 2, 1, 9223372036854775807);  view_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_633: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_34, [16, 16, 512, 1023]);  slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_18: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_16: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_633, [None, None, None, iota_18]);  view_633 = iota_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_178: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_627, index_16);  view_627 = index_16 = None
        add_179: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_178, 0);  add_178 = None

        # No stacktrace found for following nodes
        mul_tensor_28: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_179, 0.125)
        convert_element_type_default_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.float32);  mul_tensor_28 = None
        eq_tensor_7: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_14, convert_element_type_default_14)
        abs_default_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_14)
        ne_scalar_7: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_7, inf);  abs_default_7 = None
        mul_tensor_31: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_7, ne_scalar_7);  eq_tensor_7 = ne_scalar_7 = None
        logical_not_default_14: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_31);  mul_tensor_31 = None
        any_dims_7: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_14, [3], True);  logical_not_default_14 = None
        logical_not_default_15: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_7);  any_dims_7 = None
        convert_element_type_default_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.float32);  add_179 = None
        mul_tensor_29: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_15, 1);  convert_element_type_default_15 = None
        amax_default_14: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_29, [3], True)
        sub_tensor_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_29, amax_default_14);  mul_tensor_29 = amax_default_14 = None
        mul_tensor_30: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_14, 0.125);  sub_tensor_14 = None
        amax_default_15: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_14, [3], True)
        sub_tensor_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_14, amax_default_15);  convert_element_type_default_14 = amax_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_15, mul_tensor_30, sub_tensor_15);  logical_not_default_15 = mul_tensor_30 = sub_tensor_15 = None
        exp_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_7);  where_self_7 = None
        sum_17: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [3], True)
        div_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_512: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_423: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_512, 4);  convert_element_type_512 = None
        view_634: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_423, [256, 512, 512]);  unsqueeze_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_411: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_497, 3)
        unsqueeze_412: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 4);  unsqueeze_411 = None
        view_616: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_412, [1, 8192, 1024]);  unsqueeze_412 = None
        squeeze_dim_74: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_616, 0);  view_616 = None
        unsqueeze_413: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg244_1, 3);  arg244_1 = None
        unsqueeze_414: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 4);  unsqueeze_413 = None
        view_617: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_414, [1, 1024, 1024]);  unsqueeze_414 = None
        squeeze_dim_75: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_617, 0);  view_617 = None
        mm_default_37: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_74, squeeze_dim_75);  squeeze_dim_74 = squeeze_dim_75 = None
        unsqueeze_default_37: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_37, 0);  mm_default_37 = None
        view_618: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_37, [512, 16, 1, 16, 64]);  unsqueeze_default_37 = None
        permute_689: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_618, [0, 1, 3, 4, 2]);  view_618 = None
        view_619: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_689, [512, 16, 16, 64]);  permute_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_424: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_619, 4);  view_619 = None
        permute_706: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_424, [4, 1, 2, 3, 0]);  unsqueeze_424 = None
        permute_708: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_706, [1, 2, 4, 3, 0]);  permute_706 = None
        view_635: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_708, [256, 512, 64]);  permute_708 = None
        bmm_134: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_634, view_635);  view_634 = view_635 = None
        view_636: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_134, [16, 16, 512, 1, 64]);  bmm_134 = None
        permute_709: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_636, [2, 0, 1, 4, 3]);  view_636 = None
        view_637: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_709, [512, 16, 16, 64]);  permute_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_425: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_637, 4);  view_637 = None
        permute_710: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_425, [0, 1, 4, 3, 2]);  unsqueeze_425 = None
        permute_712: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_710, [0, 1, 3, 4, 2]);  permute_710 = None
        clone_100: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
        view_638: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [1, 8192, 1024]);  clone_100 = None
        squeeze_dim_70: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_638, 0);  view_638 = None
        unsqueeze_426: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg248_1, 3);  arg248_1 = None
        unsqueeze_427: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 4);  unsqueeze_426 = None
        permute_711: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_427, [3, 4, 0, 2, 1]);  unsqueeze_427 = None
        permute_713: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_711, [3, 4, 2, 0, 1]);  permute_711 = None
        clone_101: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_713, memory_format = torch.contiguous_format);  permute_713 = None
        view_639: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [1, 1024, 1024]);  clone_101 = None
        squeeze_dim_71: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_639, 0);  view_639 = None
        mm_default_35: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_70, squeeze_dim_71);  squeeze_dim_70 = squeeze_dim_71 = None
        unsqueeze_default_35: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_35, 0);  mm_default_35 = None
        view_640: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_35, [512, 16, 1, 1, 1024]);  unsqueeze_default_35 = None
        permute_714: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_640, [0, 1, 4, 2, 3]);  view_640 = None
        view_641: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_714, [512, 16, 1024]);  permute_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_180: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_641, convert_element_type_497);  view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_517: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_180, torch.float32);  add_180 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_517, [2], correction = 0, keepdim = True)
        getitem_64: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        sub_49: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_517, getitem_65);  convert_element_type_517 = getitem_65 = None
        add_181: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        mul_131: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
        mul_132: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, arg249_1);  mul_131 = arg249_1 = None
        add_182: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_132, arg250_1);  mul_132 = arg250_1 = None
        convert_element_type_518: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_182, torch.bfloat16);  add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_642: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_518, [8192, 1024])
        permute_715: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_32: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg252_1, view_642, permute_715);  arg252_1 = view_642 = permute_715 = None
        view_643: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [512, 16, 4096]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_522: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_643, torch.float32);  view_643 = None
        mul_133: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_522, 0.5)
        mul_134: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_522, 0.7071067811865476);  convert_element_type_522 = None
        erf_16: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
        add_183: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_135: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, add_183);  mul_133 = add_183 = None
        convert_element_type_523: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_644: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_523, [8192, 4096]);  convert_element_type_523 = None
        permute_716: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_33: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg254_1, view_644, permute_716);  arg254_1 = view_644 = permute_716 = None
        view_645: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [512, 16, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_184: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_645, convert_element_type_518);  view_645 = convert_element_type_518 = None
        convert_element_type_527: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.float32);  add_184 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_527, [2], correction = 0, keepdim = True)
        getitem_66: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        sub_50: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_527, getitem_67);  convert_element_type_527 = getitem_67 = None
        add_185: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        mul_136: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
        mul_137: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, arg255_1);  mul_136 = arg255_1 = None
        add_186: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, arg256_1);  mul_137 = arg256_1 = None
        convert_element_type_528: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16);  add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_428: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_528, 3)
        unsqueeze_429: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 4);  unsqueeze_428 = None
        view_646: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_429, [1, 8192, 1024]);  unsqueeze_429 = None
        squeeze_dim_68: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_646, 0);  view_646 = None
        unsqueeze_430: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg257_1, 3);  arg257_1 = None
        unsqueeze_431: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 4);  unsqueeze_430 = None
        view_647: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_431, [1, 1024, 1024]);  unsqueeze_431 = None
        squeeze_dim_69: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_647, 0);  view_647 = None
        mm_default_34: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_68, squeeze_dim_69);  squeeze_dim_68 = squeeze_dim_69 = None
        unsqueeze_default_34: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_34, 0);  mm_default_34 = None
        view_648: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_34, [512, 16, 1, 16, 64]);  unsqueeze_default_34 = None
        permute_721: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_648, [0, 1, 3, 4, 2]);  view_648 = None
        view_649: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_721, [512, 16, 16, 64]);  permute_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_187: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_649, arg261_1);  arg261_1 = None
        unsqueeze_444: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_187, 4);  add_187 = None
        permute_737: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_444, [1, 2, 0, 4, 3]);  unsqueeze_444 = None
        permute_739: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_737, [0, 1, 2, 4, 3]);  permute_737 = None
        view_662: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_739, [256, 512, 64]);  permute_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_432: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_528, 3)
        unsqueeze_433: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 4);  unsqueeze_432 = None
        view_650: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_433, [1, 8192, 1024]);  unsqueeze_433 = None
        squeeze_dim_66: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_650, 0);  view_650 = None
        unsqueeze_434: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg258_1, 3);  arg258_1 = None
        unsqueeze_435: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 4);  unsqueeze_434 = None
        view_651: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_435, [1, 1024, 1024]);  unsqueeze_435 = None
        squeeze_dim_67: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_651, 0);  view_651 = None
        mm_default_33: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_66, squeeze_dim_67);  squeeze_dim_66 = squeeze_dim_67 = None
        unsqueeze_default_33: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_33, 0);  mm_default_33 = None
        view_652: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_33, [512, 16, 1, 16, 64]);  unsqueeze_default_33 = None
        permute_726: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_652, [0, 1, 3, 4, 2]);  view_652 = None
        view_653: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_726, [512, 16, 16, 64]);  permute_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_445: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_653, 4);  view_653 = None
        permute_738: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_445, [1, 2, 4, 0, 3]);  unsqueeze_445 = None
        permute_740: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_738, [0, 1, 4, 3, 2]);  permute_738 = None
        view_663: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_740, [256, 64, 512]);  permute_740 = None
        bmm_140: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_662, view_663);  view_662 = view_663 = None
        view_664: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_140, [16, 16, 512, 1, 512]);  bmm_140 = None
        permute_741: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_664, [0, 1, 2, 4, 3]);  view_664 = None
        view_665: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_741, [16, 16, 512, 512]);  permute_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_188: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_649, arg262_1);  view_649 = arg262_1 = None
        unsqueeze_446: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_188, 4);  add_188 = None
        permute_742: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_446, [1, 2, 0, 4, 3]);  unsqueeze_446 = None
        permute_744: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_742, [0, 1, 2, 4, 3]);  permute_742 = None
        view_666: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_744, [256, 512, 64]);  permute_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_535: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_440: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_535, 3);  convert_element_type_535 = None
        unsqueeze_441: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 4);  unsqueeze_440 = None
        view_658: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_441, [1, 16384, 1024]);  unsqueeze_441 = None
        squeeze_dim_62: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_658, 0);  view_658 = None
        unsqueeze_442: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg260_1, 3);  arg260_1 = None
        unsqueeze_443: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 4);  unsqueeze_442 = None
        view_659: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_443, [1, 1024, 1024]);  unsqueeze_443 = None
        squeeze_dim_63: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_659, 0);  view_659 = None
        mm_default_31: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_62, squeeze_dim_63);  squeeze_dim_62 = squeeze_dim_63 = None
        unsqueeze_default_31: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_31, 0);  mm_default_31 = None
        view_660: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_31, [1024, 16, 1, 16, 64]);  unsqueeze_default_31 = None
        permute_736: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_660, [0, 1, 3, 4, 2]);  view_660 = None
        view_661: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_736, [1024, 16, 16, 64]);  permute_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_447: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_661, 4);  view_661 = None
        permute_743: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_447, [1, 2, 4, 0, 3]);  unsqueeze_447 = None
        permute_745: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_743, [0, 1, 4, 3, 2]);  permute_743 = None
        view_667: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_745, [256, 64, 1024]);  permute_745 = None
        bmm_141: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_666, view_667);  view_666 = view_667 = None
        view_668: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_141, [16, 16, 512, 1, 1024]);  bmm_141 = None
        permute_746: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_668, [0, 1, 2, 4, 3]);  view_668 = None
        view_669: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_746, [16, 16, 512, 1024]);  permute_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_670: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_669, [16, 16, 1024, 512]);  view_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_36: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_670, 2, 1, 9223372036854775807);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_671: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_36, [16, 16, 512, 1023]);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_19: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_17: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_671, [None, None, None, iota_19]);  view_671 = iota_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_189: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_665, index_17);  view_665 = index_17 = None
        add_190: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_189, 0);  add_189 = None

        # No stacktrace found for following nodes
        mul_tensor_24: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_190, 0.125)
        convert_element_type_default_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.float32);  mul_tensor_24 = None
        eq_tensor_6: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_12, convert_element_type_default_12)
        abs_default_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_12)
        ne_scalar_6: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_6, inf);  abs_default_6 = None
        mul_tensor_27: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_6, ne_scalar_6);  eq_tensor_6 = ne_scalar_6 = None
        logical_not_default_12: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_27);  mul_tensor_27 = None
        any_dims_6: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_12, [3], True);  logical_not_default_12 = None
        logical_not_default_13: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_6);  any_dims_6 = None
        convert_element_type_default_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_190, torch.float32);  add_190 = None
        mul_tensor_25: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_13, 1);  convert_element_type_default_13 = None
        amax_default_12: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_25, [3], True)
        sub_tensor_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_25, amax_default_12);  mul_tensor_25 = amax_default_12 = None
        mul_tensor_26: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_12, 0.125);  sub_tensor_12 = None
        amax_default_13: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_12, [3], True)
        sub_tensor_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_12, amax_default_13);  convert_element_type_default_12 = amax_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_13, mul_tensor_26, sub_tensor_13);  logical_not_default_13 = mul_tensor_26 = sub_tensor_13 = None
        exp_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_6);  where_self_6 = None
        sum_18: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [3], True)
        div_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_543: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_448: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_543, 4);  convert_element_type_543 = None
        view_672: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_448, [256, 512, 512]);  unsqueeze_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_436: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_528, 3)
        unsqueeze_437: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 4);  unsqueeze_436 = None
        view_654: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_437, [1, 8192, 1024]);  unsqueeze_437 = None
        squeeze_dim_64: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_654, 0);  view_654 = None
        unsqueeze_438: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg259_1, 3);  arg259_1 = None
        unsqueeze_439: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 4);  unsqueeze_438 = None
        view_655: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_439, [1, 1024, 1024]);  unsqueeze_439 = None
        squeeze_dim_65: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_655, 0);  view_655 = None
        mm_default_32: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_64, squeeze_dim_65);  squeeze_dim_64 = squeeze_dim_65 = None
        unsqueeze_default_32: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_32, 0);  mm_default_32 = None
        view_656: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_32, [512, 16, 1, 16, 64]);  unsqueeze_default_32 = None
        permute_731: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_656, [0, 1, 3, 4, 2]);  view_656 = None
        view_657: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_731, [512, 16, 16, 64]);  permute_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_449: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_657, 4);  view_657 = None
        permute_748: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_449, [4, 1, 2, 3, 0]);  unsqueeze_449 = None
        permute_750: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_748, [1, 2, 4, 3, 0]);  permute_748 = None
        view_673: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_750, [256, 512, 64]);  permute_750 = None
        bmm_142: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_672, view_673);  view_672 = view_673 = None
        view_674: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_142, [16, 16, 512, 1, 64]);  bmm_142 = None
        permute_751: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_674, [2, 0, 1, 4, 3]);  view_674 = None
        view_675: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_751, [512, 16, 16, 64]);  permute_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_450: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_675, 4);  view_675 = None
        permute_752: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_450, [0, 1, 4, 3, 2]);  unsqueeze_450 = None
        permute_754: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_752, [0, 1, 3, 4, 2]);  permute_752 = None
        clone_106: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
        view_676: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [1, 8192, 1024]);  clone_106 = None
        squeeze_dim_60: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_676, 0);  view_676 = None
        unsqueeze_451: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg263_1, 3);  arg263_1 = None
        unsqueeze_452: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 4);  unsqueeze_451 = None
        permute_753: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_452, [3, 4, 0, 2, 1]);  unsqueeze_452 = None
        permute_755: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_753, [3, 4, 2, 0, 1]);  permute_753 = None
        clone_107: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_755, memory_format = torch.contiguous_format);  permute_755 = None
        view_677: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [1, 1024, 1024]);  clone_107 = None
        squeeze_dim_61: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_677, 0);  view_677 = None
        mm_default_30: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_60, squeeze_dim_61);  squeeze_dim_60 = squeeze_dim_61 = None
        unsqueeze_default_30: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_30, 0);  mm_default_30 = None
        view_678: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_30, [512, 16, 1, 1, 1024]);  unsqueeze_default_30 = None
        permute_756: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_678, [0, 1, 4, 2, 3]);  view_678 = None
        view_679: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_756, [512, 16, 1024]);  permute_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_191: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_679, convert_element_type_528);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_548: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.float32);  add_191 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_548, [2], correction = 0, keepdim = True)
        getitem_68: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        sub_52: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_548, getitem_69);  convert_element_type_548 = getitem_69 = None
        add_192: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_139: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
        mul_140: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, arg264_1);  mul_139 = arg264_1 = None
        add_193: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_140, arg265_1);  mul_140 = arg265_1 = None
        convert_element_type_549: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16);  add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_680: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_549, [8192, 1024])
        permute_757: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_34: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg267_1, view_680, permute_757);  arg267_1 = view_680 = permute_757 = None
        view_681: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [512, 16, 4096]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_553: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_681, torch.float32);  view_681 = None
        mul_141: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_553, 0.5)
        mul_142: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_553, 0.7071067811865476);  convert_element_type_553 = None
        erf_17: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_194: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_143: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, add_194);  mul_141 = add_194 = None
        convert_element_type_554: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_682: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_554, [8192, 4096]);  convert_element_type_554 = None
        permute_758: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_35: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg269_1, view_682, permute_758);  arg269_1 = view_682 = permute_758 = None
        view_683: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [512, 16, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_195: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_683, convert_element_type_549);  view_683 = convert_element_type_549 = None
        convert_element_type_558: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_195, torch.float32);  add_195 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_558, [2], correction = 0, keepdim = True)
        getitem_70: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        sub_53: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_558, getitem_71);  convert_element_type_558 = getitem_71 = None
        add_196: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        mul_144: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
        mul_145: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, arg270_1);  mul_144 = arg270_1 = None
        add_197: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_145, arg271_1);  mul_145 = arg271_1 = None
        convert_element_type_559: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16);  add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_453: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_559, 3)
        unsqueeze_454: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 4);  unsqueeze_453 = None
        view_684: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_454, [1, 8192, 1024]);  unsqueeze_454 = None
        squeeze_dim_58: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_684, 0);  view_684 = None
        unsqueeze_455: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg272_1, 3);  arg272_1 = None
        unsqueeze_456: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 4);  unsqueeze_455 = None
        view_685: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_456, [1, 1024, 1024]);  unsqueeze_456 = None
        squeeze_dim_59: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_685, 0);  view_685 = None
        mm_default_29: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_58, squeeze_dim_59);  squeeze_dim_58 = squeeze_dim_59 = None
        unsqueeze_default_29: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_29, 0);  mm_default_29 = None
        view_686: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_29, [512, 16, 1, 16, 64]);  unsqueeze_default_29 = None
        permute_763: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_686, [0, 1, 3, 4, 2]);  view_686 = None
        view_687: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_763, [512, 16, 16, 64]);  permute_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_198: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_687, arg276_1);  arg276_1 = None
        unsqueeze_469: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_198, 4);  add_198 = None
        permute_779: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_469, [1, 2, 0, 4, 3]);  unsqueeze_469 = None
        permute_781: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_779, [0, 1, 2, 4, 3]);  permute_779 = None
        view_700: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_781, [256, 512, 64]);  permute_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_457: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_559, 3)
        unsqueeze_458: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 4);  unsqueeze_457 = None
        view_688: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_458, [1, 8192, 1024]);  unsqueeze_458 = None
        squeeze_dim_56: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_688, 0);  view_688 = None
        unsqueeze_459: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg273_1, 3);  arg273_1 = None
        unsqueeze_460: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 4);  unsqueeze_459 = None
        view_689: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_460, [1, 1024, 1024]);  unsqueeze_460 = None
        squeeze_dim_57: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_689, 0);  view_689 = None
        mm_default_28: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_56, squeeze_dim_57);  squeeze_dim_56 = squeeze_dim_57 = None
        unsqueeze_default_28: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_28, 0);  mm_default_28 = None
        view_690: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_28, [512, 16, 1, 16, 64]);  unsqueeze_default_28 = None
        permute_768: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_690, [0, 1, 3, 4, 2]);  view_690 = None
        view_691: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_768, [512, 16, 16, 64]);  permute_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_470: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_691, 4);  view_691 = None
        permute_780: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_470, [1, 2, 4, 0, 3]);  unsqueeze_470 = None
        permute_782: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_780, [0, 1, 4, 3, 2]);  permute_780 = None
        view_701: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_782, [256, 64, 512]);  permute_782 = None
        bmm_148: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_700, view_701);  view_700 = view_701 = None
        view_702: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_148, [16, 16, 512, 1, 512]);  bmm_148 = None
        permute_783: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_702, [0, 1, 2, 4, 3]);  view_702 = None
        view_703: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_783, [16, 16, 512, 512]);  permute_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_199: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_687, arg277_1);  view_687 = arg277_1 = None
        unsqueeze_471: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_199, 4);  add_199 = None
        permute_784: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_471, [1, 2, 0, 4, 3]);  unsqueeze_471 = None
        permute_786: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_784, [0, 1, 2, 4, 3]);  permute_784 = None
        view_704: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_786, [256, 512, 64]);  permute_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_566: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_465: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_566, 3);  convert_element_type_566 = None
        unsqueeze_466: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 4);  unsqueeze_465 = None
        view_696: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_466, [1, 16384, 1024]);  unsqueeze_466 = None
        squeeze_dim_52: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_696, 0);  view_696 = None
        unsqueeze_467: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg275_1, 3);  arg275_1 = None
        unsqueeze_468: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 4);  unsqueeze_467 = None
        view_697: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_468, [1, 1024, 1024]);  unsqueeze_468 = None
        squeeze_dim_53: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_697, 0);  view_697 = None
        mm_default_26: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_52, squeeze_dim_53);  squeeze_dim_52 = squeeze_dim_53 = None
        unsqueeze_default_26: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_26, 0);  mm_default_26 = None
        view_698: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_26, [1024, 16, 1, 16, 64]);  unsqueeze_default_26 = None
        permute_778: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_698, [0, 1, 3, 4, 2]);  view_698 = None
        view_699: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_778, [1024, 16, 16, 64]);  permute_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_472: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_699, 4);  view_699 = None
        permute_785: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_472, [1, 2, 4, 0, 3]);  unsqueeze_472 = None
        permute_787: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_785, [0, 1, 4, 3, 2]);  permute_785 = None
        view_705: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_787, [256, 64, 1024]);  permute_787 = None
        bmm_149: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_704, view_705);  view_704 = view_705 = None
        view_706: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_149, [16, 16, 512, 1, 1024]);  bmm_149 = None
        permute_788: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_706, [0, 1, 2, 4, 3]);  view_706 = None
        view_707: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_788, [16, 16, 512, 1024]);  permute_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_708: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_707, [16, 16, 1024, 512]);  view_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_38: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_708, 2, 1, 9223372036854775807);  view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_709: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_38, [16, 16, 512, 1023]);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_20: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_18: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_709, [None, None, None, iota_20]);  view_709 = iota_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_200: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_703, index_18);  view_703 = index_18 = None
        add_201: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_200, 0);  add_200 = None

        # No stacktrace found for following nodes
        mul_tensor_20: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_201, 0.125)
        convert_element_type_default_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.float32);  mul_tensor_20 = None
        eq_tensor_5: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_10, convert_element_type_default_10)
        abs_default_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_10)
        ne_scalar_5: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_5, inf);  abs_default_5 = None
        mul_tensor_23: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_5, ne_scalar_5);  eq_tensor_5 = ne_scalar_5 = None
        logical_not_default_10: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_23);  mul_tensor_23 = None
        any_dims_5: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_10, [3], True);  logical_not_default_10 = None
        logical_not_default_11: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_5);  any_dims_5 = None
        convert_element_type_default_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_201, torch.float32);  add_201 = None
        mul_tensor_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1);  convert_element_type_default_11 = None
        amax_default_10: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_21, [3], True)
        sub_tensor_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_21, amax_default_10);  mul_tensor_21 = amax_default_10 = None
        mul_tensor_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_10, 0.125);  sub_tensor_10 = None
        amax_default_11: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_10, [3], True)
        sub_tensor_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_10, amax_default_11);  convert_element_type_default_10 = amax_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_11, mul_tensor_22, sub_tensor_11);  logical_not_default_11 = mul_tensor_22 = sub_tensor_11 = None
        exp_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_5);  where_self_5 = None
        sum_19: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [3], True)
        div_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_574: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_473: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_574, 4);  convert_element_type_574 = None
        view_710: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_473, [256, 512, 512]);  unsqueeze_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_461: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_559, 3)
        unsqueeze_462: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 4);  unsqueeze_461 = None
        view_692: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_462, [1, 8192, 1024]);  unsqueeze_462 = None
        squeeze_dim_54: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_692, 0);  view_692 = None
        unsqueeze_463: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg274_1, 3);  arg274_1 = None
        unsqueeze_464: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 4);  unsqueeze_463 = None
        view_693: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_464, [1, 1024, 1024]);  unsqueeze_464 = None
        squeeze_dim_55: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_693, 0);  view_693 = None
        mm_default_27: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_54, squeeze_dim_55);  squeeze_dim_54 = squeeze_dim_55 = None
        unsqueeze_default_27: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_27, 0);  mm_default_27 = None
        view_694: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_27, [512, 16, 1, 16, 64]);  unsqueeze_default_27 = None
        permute_773: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_694, [0, 1, 3, 4, 2]);  view_694 = None
        view_695: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_773, [512, 16, 16, 64]);  permute_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_474: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_695, 4);  view_695 = None
        permute_790: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_474, [4, 1, 2, 3, 0]);  unsqueeze_474 = None
        permute_792: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_790, [1, 2, 4, 3, 0]);  permute_790 = None
        view_711: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_792, [256, 512, 64]);  permute_792 = None
        bmm_150: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_710, view_711);  view_710 = view_711 = None
        view_712: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_150, [16, 16, 512, 1, 64]);  bmm_150 = None
        permute_793: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_712, [2, 0, 1, 4, 3]);  view_712 = None
        view_713: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_793, [512, 16, 16, 64]);  permute_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_475: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_713, 4);  view_713 = None
        permute_794: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_475, [0, 1, 4, 3, 2]);  unsqueeze_475 = None
        permute_796: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_794, [0, 1, 3, 4, 2]);  permute_794 = None
        clone_112: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_796, memory_format = torch.contiguous_format);  permute_796 = None
        view_714: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_112, [1, 8192, 1024]);  clone_112 = None
        squeeze_dim_50: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_714, 0);  view_714 = None
        unsqueeze_476: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg278_1, 3);  arg278_1 = None
        unsqueeze_477: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 4);  unsqueeze_476 = None
        permute_795: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_477, [3, 4, 0, 2, 1]);  unsqueeze_477 = None
        permute_797: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_795, [3, 4, 2, 0, 1]);  permute_795 = None
        clone_113: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format);  permute_797 = None
        view_715: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [1, 1024, 1024]);  clone_113 = None
        squeeze_dim_51: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_715, 0);  view_715 = None
        mm_default_25: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_50, squeeze_dim_51);  squeeze_dim_50 = squeeze_dim_51 = None
        unsqueeze_default_25: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_25, 0);  mm_default_25 = None
        view_716: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_25, [512, 16, 1, 1, 1024]);  unsqueeze_default_25 = None
        permute_798: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_716, [0, 1, 4, 2, 3]);  view_716 = None
        view_717: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_798, [512, 16, 1024]);  permute_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_202: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_717, convert_element_type_559);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_579: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.float32);  add_202 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_579, [2], correction = 0, keepdim = True)
        getitem_72: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        sub_55: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_579, getitem_73);  convert_element_type_579 = getitem_73 = None
        add_203: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_147: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = rsqrt_36 = None
        mul_148: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, arg279_1);  mul_147 = arg279_1 = None
        add_204: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_148, arg280_1);  mul_148 = arg280_1 = None
        convert_element_type_580: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_718: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_580, [8192, 1024])
        permute_799: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_36: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg282_1, view_718, permute_799);  arg282_1 = view_718 = permute_799 = None
        view_719: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [512, 16, 4096]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_584: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_719, torch.float32);  view_719 = None
        mul_149: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_584, 0.5)
        mul_150: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_584, 0.7071067811865476);  convert_element_type_584 = None
        erf_18: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_150);  mul_150 = None
        add_205: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_151: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_149, add_205);  mul_149 = add_205 = None
        convert_element_type_585: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_151, torch.bfloat16);  mul_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_720: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_585, [8192, 4096]);  convert_element_type_585 = None
        permute_800: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_37: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg284_1, view_720, permute_800);  arg284_1 = view_720 = permute_800 = None
        view_721: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [512, 16, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_206: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_721, convert_element_type_580);  view_721 = convert_element_type_580 = None
        convert_element_type_589: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_206, torch.float32);  add_206 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_589, [2], correction = 0, keepdim = True)
        getitem_74: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        sub_56: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_589, getitem_75);  convert_element_type_589 = getitem_75 = None
        add_207: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        mul_152: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
        mul_153: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, arg285_1);  mul_152 = arg285_1 = None
        add_208: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_153, arg286_1);  mul_153 = arg286_1 = None
        convert_element_type_590: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_478: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_590, 3)
        unsqueeze_479: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 4);  unsqueeze_478 = None
        view_722: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_479, [1, 8192, 1024]);  unsqueeze_479 = None
        squeeze_dim_48: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_722, 0);  view_722 = None
        unsqueeze_480: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg287_1, 3);  arg287_1 = None
        unsqueeze_481: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 4);  unsqueeze_480 = None
        view_723: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_481, [1, 1024, 1024]);  unsqueeze_481 = None
        squeeze_dim_49: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_723, 0);  view_723 = None
        mm_default_24: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_48, squeeze_dim_49);  squeeze_dim_48 = squeeze_dim_49 = None
        unsqueeze_default_24: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_24, 0);  mm_default_24 = None
        view_724: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_24, [512, 16, 1, 16, 64]);  unsqueeze_default_24 = None
        permute_805: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_724, [0, 1, 3, 4, 2]);  view_724 = None
        view_725: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_805, [512, 16, 16, 64]);  permute_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_209: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_725, arg291_1);  arg291_1 = None
        unsqueeze_494: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_209, 4);  add_209 = None
        permute_821: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_494, [1, 2, 0, 4, 3]);  unsqueeze_494 = None
        permute_823: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_821, [0, 1, 2, 4, 3]);  permute_821 = None
        view_738: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_823, [256, 512, 64]);  permute_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_482: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_590, 3)
        unsqueeze_483: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 4);  unsqueeze_482 = None
        view_726: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_483, [1, 8192, 1024]);  unsqueeze_483 = None
        squeeze_dim_46: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_726, 0);  view_726 = None
        unsqueeze_484: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg288_1, 3);  arg288_1 = None
        unsqueeze_485: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 4);  unsqueeze_484 = None
        view_727: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_485, [1, 1024, 1024]);  unsqueeze_485 = None
        squeeze_dim_47: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_727, 0);  view_727 = None
        mm_default_23: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_46, squeeze_dim_47);  squeeze_dim_46 = squeeze_dim_47 = None
        unsqueeze_default_23: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_23, 0);  mm_default_23 = None
        view_728: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_23, [512, 16, 1, 16, 64]);  unsqueeze_default_23 = None
        permute_810: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_728, [0, 1, 3, 4, 2]);  view_728 = None
        view_729: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_810, [512, 16, 16, 64]);  permute_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_495: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_729, 4);  view_729 = None
        permute_822: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_495, [1, 2, 4, 0, 3]);  unsqueeze_495 = None
        permute_824: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_822, [0, 1, 4, 3, 2]);  permute_822 = None
        view_739: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_824, [256, 64, 512]);  permute_824 = None
        bmm_156: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_738, view_739);  view_738 = view_739 = None
        view_740: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_156, [16, 16, 512, 1, 512]);  bmm_156 = None
        permute_825: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_740, [0, 1, 2, 4, 3]);  view_740 = None
        view_741: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_825, [16, 16, 512, 512]);  permute_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_210: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_725, arg292_1);  view_725 = arg292_1 = None
        unsqueeze_496: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_210, 4);  add_210 = None
        permute_826: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_496, [1, 2, 0, 4, 3]);  unsqueeze_496 = None
        permute_828: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_826, [0, 1, 2, 4, 3]);  permute_826 = None
        view_742: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_828, [256, 512, 64]);  permute_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_597: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_490: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_597, 3);  convert_element_type_597 = None
        unsqueeze_491: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 4);  unsqueeze_490 = None
        view_734: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_491, [1, 16384, 1024]);  unsqueeze_491 = None
        squeeze_dim_42: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_734, 0);  view_734 = None
        unsqueeze_492: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg290_1, 3);  arg290_1 = None
        unsqueeze_493: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 4);  unsqueeze_492 = None
        view_735: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_493, [1, 1024, 1024]);  unsqueeze_493 = None
        squeeze_dim_43: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_735, 0);  view_735 = None
        mm_default_21: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_42, squeeze_dim_43);  squeeze_dim_42 = squeeze_dim_43 = None
        unsqueeze_default_21: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_21, 0);  mm_default_21 = None
        view_736: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_21, [1024, 16, 1, 16, 64]);  unsqueeze_default_21 = None
        permute_820: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_736, [0, 1, 3, 4, 2]);  view_736 = None
        view_737: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_820, [1024, 16, 16, 64]);  permute_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_497: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_737, 4);  view_737 = None
        permute_827: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_497, [1, 2, 4, 0, 3]);  unsqueeze_497 = None
        permute_829: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_827, [0, 1, 4, 3, 2]);  permute_827 = None
        view_743: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_829, [256, 64, 1024]);  permute_829 = None
        bmm_157: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_742, view_743);  view_742 = view_743 = None
        view_744: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_157, [16, 16, 512, 1, 1024]);  bmm_157 = None
        permute_830: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_744, [0, 1, 2, 4, 3]);  view_744 = None
        view_745: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_830, [16, 16, 512, 1024]);  permute_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_746: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_745, [16, 16, 1024, 512]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_40: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_746, 2, 1, 9223372036854775807);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_747: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_40, [16, 16, 512, 1023]);  slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_21: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_19: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_747, [None, None, None, iota_21]);  view_747 = iota_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_211: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_741, index_19);  view_741 = index_19 = None
        add_212: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_211, 0);  add_211 = None

        # No stacktrace found for following nodes
        mul_tensor_16: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_212, 0.125)
        convert_element_type_default_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.float32);  mul_tensor_16 = None
        eq_tensor_4: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_8, convert_element_type_default_8)
        abs_default_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_8)
        ne_scalar_4: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_4, inf);  abs_default_4 = None
        mul_tensor_19: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_4, ne_scalar_4);  eq_tensor_4 = ne_scalar_4 = None
        logical_not_default_8: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_19);  mul_tensor_19 = None
        any_dims_4: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_8, [3], True);  logical_not_default_8 = None
        logical_not_default_9: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_4);  any_dims_4 = None
        convert_element_type_default_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_212, torch.float32);  add_212 = None
        mul_tensor_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1);  convert_element_type_default_9 = None
        amax_default_8: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_17, [3], True)
        sub_tensor_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_17, amax_default_8);  mul_tensor_17 = amax_default_8 = None
        mul_tensor_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_8, 0.125);  sub_tensor_8 = None
        amax_default_9: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_8, [3], True)
        sub_tensor_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_8, amax_default_9);  convert_element_type_default_8 = amax_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_9, mul_tensor_18, sub_tensor_9);  logical_not_default_9 = mul_tensor_18 = sub_tensor_9 = None
        exp_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_4);  where_self_4 = None
        sum_20: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [3], True)
        div_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_605: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_498: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_605, 4);  convert_element_type_605 = None
        view_748: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_498, [256, 512, 512]);  unsqueeze_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_486: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_590, 3)
        unsqueeze_487: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 4);  unsqueeze_486 = None
        view_730: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_487, [1, 8192, 1024]);  unsqueeze_487 = None
        squeeze_dim_44: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_730, 0);  view_730 = None
        unsqueeze_488: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg289_1, 3);  arg289_1 = None
        unsqueeze_489: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 4);  unsqueeze_488 = None
        view_731: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_489, [1, 1024, 1024]);  unsqueeze_489 = None
        squeeze_dim_45: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_731, 0);  view_731 = None
        mm_default_22: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_44, squeeze_dim_45);  squeeze_dim_44 = squeeze_dim_45 = None
        unsqueeze_default_22: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_22, 0);  mm_default_22 = None
        view_732: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_22, [512, 16, 1, 16, 64]);  unsqueeze_default_22 = None
        permute_815: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_732, [0, 1, 3, 4, 2]);  view_732 = None
        view_733: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_815, [512, 16, 16, 64]);  permute_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_499: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_733, 4);  view_733 = None
        permute_832: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_499, [4, 1, 2, 3, 0]);  unsqueeze_499 = None
        permute_834: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_832, [1, 2, 4, 3, 0]);  permute_832 = None
        view_749: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_834, [256, 512, 64]);  permute_834 = None
        bmm_158: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_748, view_749);  view_748 = view_749 = None
        view_750: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_158, [16, 16, 512, 1, 64]);  bmm_158 = None
        permute_835: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_750, [2, 0, 1, 4, 3]);  view_750 = None
        view_751: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_835, [512, 16, 16, 64]);  permute_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_500: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_751, 4);  view_751 = None
        permute_836: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_500, [0, 1, 4, 3, 2]);  unsqueeze_500 = None
        permute_838: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_836, [0, 1, 3, 4, 2]);  permute_836 = None
        clone_118: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
        view_752: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [1, 8192, 1024]);  clone_118 = None
        squeeze_dim_40: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_752, 0);  view_752 = None
        unsqueeze_501: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg293_1, 3);  arg293_1 = None
        unsqueeze_502: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 4);  unsqueeze_501 = None
        permute_837: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_502, [3, 4, 0, 2, 1]);  unsqueeze_502 = None
        permute_839: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_837, [3, 4, 2, 0, 1]);  permute_837 = None
        clone_119: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_839, memory_format = torch.contiguous_format);  permute_839 = None
        view_753: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_119, [1, 1024, 1024]);  clone_119 = None
        squeeze_dim_41: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_753, 0);  view_753 = None
        mm_default_20: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_40, squeeze_dim_41);  squeeze_dim_40 = squeeze_dim_41 = None
        unsqueeze_default_20: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_20, 0);  mm_default_20 = None
        view_754: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_20, [512, 16, 1, 1, 1024]);  unsqueeze_default_20 = None
        permute_840: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_754, [0, 1, 4, 2, 3]);  view_754 = None
        view_755: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_840, [512, 16, 1024]);  permute_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_213: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_755, convert_element_type_590);  view_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_610: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_213, torch.float32);  add_213 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_610, [2], correction = 0, keepdim = True)
        getitem_76: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        sub_58: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_610, getitem_77);  convert_element_type_610 = getitem_77 = None
        add_214: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        mul_155: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = rsqrt_38 = None
        mul_156: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, arg294_1);  mul_155 = arg294_1 = None
        add_215: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_156, arg295_1);  mul_156 = arg295_1 = None
        convert_element_type_611: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_756: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_611, [8192, 1024])
        permute_841: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_38: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg297_1, view_756, permute_841);  arg297_1 = view_756 = permute_841 = None
        view_757: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [512, 16, 4096]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_615: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_757, torch.float32);  view_757 = None
        mul_157: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_615, 0.5)
        mul_158: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_615, 0.7071067811865476);  convert_element_type_615 = None
        erf_19: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_158);  mul_158 = None
        add_216: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_159: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, add_216);  mul_157 = add_216 = None
        convert_element_type_616: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_159, torch.bfloat16);  mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_758: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_616, [8192, 4096]);  convert_element_type_616 = None
        permute_842: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_39: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg299_1, view_758, permute_842);  arg299_1 = view_758 = permute_842 = None
        view_759: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [512, 16, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_217: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_759, convert_element_type_611);  view_759 = convert_element_type_611 = None
        convert_element_type_620: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_217, torch.float32);  add_217 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_620, [2], correction = 0, keepdim = True)
        getitem_78: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        sub_59: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_620, getitem_79);  convert_element_type_620 = getitem_79 = None
        add_218: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        mul_160: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
        mul_161: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, arg300_1);  mul_160 = arg300_1 = None
        add_219: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, arg301_1);  mul_161 = arg301_1 = None
        convert_element_type_621: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_503: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_621, 3)
        unsqueeze_504: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 4);  unsqueeze_503 = None
        view_760: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_504, [1, 8192, 1024]);  unsqueeze_504 = None
        squeeze_dim_38: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_760, 0);  view_760 = None
        unsqueeze_505: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg302_1, 3);  arg302_1 = None
        unsqueeze_506: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 4);  unsqueeze_505 = None
        view_761: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_506, [1, 1024, 1024]);  unsqueeze_506 = None
        squeeze_dim_39: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_761, 0);  view_761 = None
        mm_default_19: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_38, squeeze_dim_39);  squeeze_dim_38 = squeeze_dim_39 = None
        unsqueeze_default_19: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_19, 0);  mm_default_19 = None
        view_762: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_19, [512, 16, 1, 16, 64]);  unsqueeze_default_19 = None
        permute_847: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_762, [0, 1, 3, 4, 2]);  view_762 = None
        view_763: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_847, [512, 16, 16, 64]);  permute_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_220: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_763, arg306_1);  arg306_1 = None
        unsqueeze_519: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_220, 4);  add_220 = None
        permute_863: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_519, [1, 2, 0, 4, 3]);  unsqueeze_519 = None
        permute_865: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_863, [0, 1, 2, 4, 3]);  permute_863 = None
        view_776: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_865, [256, 512, 64]);  permute_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_507: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_621, 3)
        unsqueeze_508: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 4);  unsqueeze_507 = None
        view_764: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_508, [1, 8192, 1024]);  unsqueeze_508 = None
        squeeze_dim_36: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_764, 0);  view_764 = None
        unsqueeze_509: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg303_1, 3);  arg303_1 = None
        unsqueeze_510: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 4);  unsqueeze_509 = None
        view_765: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_510, [1, 1024, 1024]);  unsqueeze_510 = None
        squeeze_dim_37: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_765, 0);  view_765 = None
        mm_default_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_36, squeeze_dim_37);  squeeze_dim_36 = squeeze_dim_37 = None
        unsqueeze_default_18: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_18, 0);  mm_default_18 = None
        view_766: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_18, [512, 16, 1, 16, 64]);  unsqueeze_default_18 = None
        permute_852: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_766, [0, 1, 3, 4, 2]);  view_766 = None
        view_767: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_852, [512, 16, 16, 64]);  permute_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_520: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_767, 4);  view_767 = None
        permute_864: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_520, [1, 2, 4, 0, 3]);  unsqueeze_520 = None
        permute_866: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_864, [0, 1, 4, 3, 2]);  permute_864 = None
        view_777: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_866, [256, 64, 512]);  permute_866 = None
        bmm_164: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_776, view_777);  view_776 = view_777 = None
        view_778: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_164, [16, 16, 512, 1, 512]);  bmm_164 = None
        permute_867: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_778, [0, 1, 2, 4, 3]);  view_778 = None
        view_779: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_867, [16, 16, 512, 512]);  permute_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_221: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_763, arg307_1);  view_763 = arg307_1 = None
        unsqueeze_521: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_221, 4);  add_221 = None
        permute_868: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_521, [1, 2, 0, 4, 3]);  unsqueeze_521 = None
        permute_870: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_868, [0, 1, 2, 4, 3]);  permute_868 = None
        view_780: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_870, [256, 512, 64]);  permute_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_628: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_515: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_628, 3);  convert_element_type_628 = None
        unsqueeze_516: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 4);  unsqueeze_515 = None
        view_772: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_516, [1, 16384, 1024]);  unsqueeze_516 = None
        squeeze_dim_32: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_772, 0);  view_772 = None
        unsqueeze_517: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg305_1, 3);  arg305_1 = None
        unsqueeze_518: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 4);  unsqueeze_517 = None
        view_773: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_518, [1, 1024, 1024]);  unsqueeze_518 = None
        squeeze_dim_33: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_773, 0);  view_773 = None
        mm_default_16: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_32, squeeze_dim_33);  squeeze_dim_32 = squeeze_dim_33 = None
        unsqueeze_default_16: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_16, 0);  mm_default_16 = None
        view_774: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_16, [1024, 16, 1, 16, 64]);  unsqueeze_default_16 = None
        permute_862: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_774, [0, 1, 3, 4, 2]);  view_774 = None
        view_775: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_862, [1024, 16, 16, 64]);  permute_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_522: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_775, 4);  view_775 = None
        permute_869: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_522, [1, 2, 4, 0, 3]);  unsqueeze_522 = None
        permute_871: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_869, [0, 1, 4, 3, 2]);  permute_869 = None
        view_781: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_871, [256, 64, 1024]);  permute_871 = None
        bmm_165: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
        view_782: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_165, [16, 16, 512, 1, 1024]);  bmm_165 = None
        permute_872: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_782, [0, 1, 2, 4, 3]);  view_782 = None
        view_783: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_872, [16, 16, 512, 1024]);  permute_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_784: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_783, [16, 16, 1024, 512]);  view_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_42: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_784, 2, 1, 9223372036854775807);  view_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_785: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_42, [16, 16, 512, 1023]);  slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_22: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_20: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_785, [None, None, None, iota_22]);  view_785 = iota_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_222: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_779, index_20);  view_779 = index_20 = None
        add_223: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_222, 0);  add_222 = None

        # No stacktrace found for following nodes
        mul_tensor_12: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_223, 0.125)
        convert_element_type_default_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.float32);  mul_tensor_12 = None
        eq_tensor_3: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_6, convert_element_type_default_6)
        abs_default_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_6)
        ne_scalar_3: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_3, inf);  abs_default_3 = None
        mul_tensor_15: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_3, ne_scalar_3);  eq_tensor_3 = ne_scalar_3 = None
        logical_not_default_6: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_15);  mul_tensor_15 = None
        any_dims_3: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_6, [3], True);  logical_not_default_6 = None
        logical_not_default_7: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_3);  any_dims_3 = None
        convert_element_type_default_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_223, torch.float32);  add_223 = None
        mul_tensor_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1);  convert_element_type_default_7 = None
        amax_default_6: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_13, [3], True)
        sub_tensor_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_13, amax_default_6);  mul_tensor_13 = amax_default_6 = None
        mul_tensor_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.125);  sub_tensor_6 = None
        amax_default_7: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_6, [3], True)
        sub_tensor_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_6, amax_default_7);  convert_element_type_default_6 = amax_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_7, mul_tensor_14, sub_tensor_7);  logical_not_default_7 = mul_tensor_14 = sub_tensor_7 = None
        exp_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_3);  where_self_3 = None
        sum_21: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [3], True)
        div_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_636: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_523: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_636, 4);  convert_element_type_636 = None
        view_786: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_523, [256, 512, 512]);  unsqueeze_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_511: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_621, 3)
        unsqueeze_512: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 4);  unsqueeze_511 = None
        view_768: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_512, [1, 8192, 1024]);  unsqueeze_512 = None
        squeeze_dim_34: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_768, 0);  view_768 = None
        unsqueeze_513: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg304_1, 3);  arg304_1 = None
        unsqueeze_514: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 4);  unsqueeze_513 = None
        view_769: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_514, [1, 1024, 1024]);  unsqueeze_514 = None
        squeeze_dim_35: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_769, 0);  view_769 = None
        mm_default_17: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_34, squeeze_dim_35);  squeeze_dim_34 = squeeze_dim_35 = None
        unsqueeze_default_17: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_17, 0);  mm_default_17 = None
        view_770: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_17, [512, 16, 1, 16, 64]);  unsqueeze_default_17 = None
        permute_857: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_770, [0, 1, 3, 4, 2]);  view_770 = None
        view_771: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_857, [512, 16, 16, 64]);  permute_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_524: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_771, 4);  view_771 = None
        permute_874: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_524, [4, 1, 2, 3, 0]);  unsqueeze_524 = None
        permute_876: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_874, [1, 2, 4, 3, 0]);  permute_874 = None
        view_787: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_876, [256, 512, 64]);  permute_876 = None
        bmm_166: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_786, view_787);  view_786 = view_787 = None
        view_788: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_166, [16, 16, 512, 1, 64]);  bmm_166 = None
        permute_877: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_788, [2, 0, 1, 4, 3]);  view_788 = None
        view_789: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_877, [512, 16, 16, 64]);  permute_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_525: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_789, 4);  view_789 = None
        permute_878: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_525, [0, 1, 4, 3, 2]);  unsqueeze_525 = None
        permute_880: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_878, [0, 1, 3, 4, 2]);  permute_878 = None
        clone_124: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
        view_790: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [1, 8192, 1024]);  clone_124 = None
        squeeze_dim_30: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_790, 0);  view_790 = None
        unsqueeze_526: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg308_1, 3);  arg308_1 = None
        unsqueeze_527: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 4);  unsqueeze_526 = None
        permute_879: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_527, [3, 4, 0, 2, 1]);  unsqueeze_527 = None
        permute_881: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_879, [3, 4, 2, 0, 1]);  permute_879 = None
        clone_125: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
        view_791: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_125, [1, 1024, 1024]);  clone_125 = None
        squeeze_dim_31: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_791, 0);  view_791 = None
        mm_default_15: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_30, squeeze_dim_31);  squeeze_dim_30 = squeeze_dim_31 = None
        unsqueeze_default_15: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_15, 0);  mm_default_15 = None
        view_792: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_15, [512, 16, 1, 1, 1024]);  unsqueeze_default_15 = None
        permute_882: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_792, [0, 1, 4, 2, 3]);  view_792 = None
        view_793: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_882, [512, 16, 1024]);  permute_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_224: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_793, convert_element_type_621);  view_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_641: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_224, torch.float32);  add_224 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_641, [2], correction = 0, keepdim = True)
        getitem_80: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        sub_61: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_641, getitem_81);  convert_element_type_641 = getitem_81 = None
        add_225: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_163: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = rsqrt_40 = None
        mul_164: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, arg309_1);  mul_163 = arg309_1 = None
        add_226: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_164, arg310_1);  mul_164 = arg310_1 = None
        convert_element_type_642: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_794: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_642, [8192, 1024])
        permute_883: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_40: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg312_1, view_794, permute_883);  arg312_1 = view_794 = permute_883 = None
        view_795: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [512, 16, 4096]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_646: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_795, torch.float32);  view_795 = None
        mul_165: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_646, 0.5)
        mul_166: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_646, 0.7071067811865476);  convert_element_type_646 = None
        erf_20: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
        add_227: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_167: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, add_227);  mul_165 = add_227 = None
        convert_element_type_647: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_167, torch.bfloat16);  mul_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_796: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_647, [8192, 4096]);  convert_element_type_647 = None
        permute_884: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_41: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg314_1, view_796, permute_884);  arg314_1 = view_796 = permute_884 = None
        view_797: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [512, 16, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_228: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_797, convert_element_type_642);  view_797 = convert_element_type_642 = None
        convert_element_type_651: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.float32);  add_228 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_651, [2], correction = 0, keepdim = True)
        getitem_82: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        sub_62: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_651, getitem_83);  convert_element_type_651 = getitem_83 = None
        add_229: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_168: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
        mul_169: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, arg315_1);  mul_168 = arg315_1 = None
        add_230: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, arg316_1);  mul_169 = arg316_1 = None
        convert_element_type_652: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_230, torch.bfloat16);  add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_528: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_652, 3)
        unsqueeze_529: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 4);  unsqueeze_528 = None
        view_798: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_529, [1, 8192, 1024]);  unsqueeze_529 = None
        squeeze_dim_28: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_798, 0);  view_798 = None
        unsqueeze_530: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg317_1, 3);  arg317_1 = None
        unsqueeze_531: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 4);  unsqueeze_530 = None
        view_799: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_531, [1, 1024, 1024]);  unsqueeze_531 = None
        squeeze_dim_29: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_799, 0);  view_799 = None
        mm_default_14: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_28, squeeze_dim_29);  squeeze_dim_28 = squeeze_dim_29 = None
        unsqueeze_default_14: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_14, 0);  mm_default_14 = None
        view_800: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_14, [512, 16, 1, 16, 64]);  unsqueeze_default_14 = None
        permute_889: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_800, [0, 1, 3, 4, 2]);  view_800 = None
        view_801: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_889, [512, 16, 16, 64]);  permute_889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_231: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_801, arg321_1);  arg321_1 = None
        unsqueeze_544: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_231, 4);  add_231 = None
        permute_905: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_544, [1, 2, 0, 4, 3]);  unsqueeze_544 = None
        permute_907: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_905, [0, 1, 2, 4, 3]);  permute_905 = None
        view_814: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_907, [256, 512, 64]);  permute_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_532: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_652, 3)
        unsqueeze_533: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 4);  unsqueeze_532 = None
        view_802: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_533, [1, 8192, 1024]);  unsqueeze_533 = None
        squeeze_dim_26: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_802, 0);  view_802 = None
        unsqueeze_534: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg318_1, 3);  arg318_1 = None
        unsqueeze_535: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 4);  unsqueeze_534 = None
        view_803: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_535, [1, 1024, 1024]);  unsqueeze_535 = None
        squeeze_dim_27: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_803, 0);  view_803 = None
        mm_default_13: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_26, squeeze_dim_27);  squeeze_dim_26 = squeeze_dim_27 = None
        unsqueeze_default_13: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_13, 0);  mm_default_13 = None
        view_804: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_13, [512, 16, 1, 16, 64]);  unsqueeze_default_13 = None
        permute_894: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_804, [0, 1, 3, 4, 2]);  view_804 = None
        view_805: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_894, [512, 16, 16, 64]);  permute_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_545: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_805, 4);  view_805 = None
        permute_906: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_545, [1, 2, 4, 0, 3]);  unsqueeze_545 = None
        permute_908: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_906, [0, 1, 4, 3, 2]);  permute_906 = None
        view_815: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_908, [256, 64, 512]);  permute_908 = None
        bmm_172: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_814, view_815);  view_814 = view_815 = None
        view_816: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_172, [16, 16, 512, 1, 512]);  bmm_172 = None
        permute_909: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_816, [0, 1, 2, 4, 3]);  view_816 = None
        view_817: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_909, [16, 16, 512, 512]);  permute_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_232: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_801, arg322_1);  view_801 = arg322_1 = None
        unsqueeze_546: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_232, 4);  add_232 = None
        permute_910: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_546, [1, 2, 0, 4, 3]);  unsqueeze_546 = None
        permute_912: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_910, [0, 1, 2, 4, 3]);  permute_910 = None
        view_818: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_912, [256, 512, 64]);  permute_912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_659: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_540: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_659, 3);  convert_element_type_659 = None
        unsqueeze_541: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 4);  unsqueeze_540 = None
        view_810: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_541, [1, 16384, 1024]);  unsqueeze_541 = None
        squeeze_dim_22: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_810, 0);  view_810 = None
        unsqueeze_542: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg320_1, 3);  arg320_1 = None
        unsqueeze_543: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 4);  unsqueeze_542 = None
        view_811: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_543, [1, 1024, 1024]);  unsqueeze_543 = None
        squeeze_dim_23: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_811, 0);  view_811 = None
        mm_default_11: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_22, squeeze_dim_23);  squeeze_dim_22 = squeeze_dim_23 = None
        unsqueeze_default_11: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_11, 0);  mm_default_11 = None
        view_812: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_11, [1024, 16, 1, 16, 64]);  unsqueeze_default_11 = None
        permute_904: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_812, [0, 1, 3, 4, 2]);  view_812 = None
        view_813: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_904, [1024, 16, 16, 64]);  permute_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_547: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_813, 4);  view_813 = None
        permute_911: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_547, [1, 2, 4, 0, 3]);  unsqueeze_547 = None
        permute_913: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_911, [0, 1, 4, 3, 2]);  permute_911 = None
        view_819: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_913, [256, 64, 1024]);  permute_913 = None
        bmm_173: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_818, view_819);  view_818 = view_819 = None
        view_820: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_173, [16, 16, 512, 1, 1024]);  bmm_173 = None
        permute_914: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_820, [0, 1, 2, 4, 3]);  view_820 = None
        view_821: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_914, [16, 16, 512, 1024]);  permute_914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_822: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_821, [16, 16, 1024, 512]);  view_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_44: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_822, 2, 1, 9223372036854775807);  view_822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_823: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_44, [16, 16, 512, 1023]);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_23: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_21: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_823, [None, None, None, iota_23]);  view_823 = iota_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_233: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_817, index_21);  view_817 = index_21 = None
        add_234: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_233, 0);  add_233 = None

        # No stacktrace found for following nodes
        mul_tensor_8: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_234, 0.125)
        convert_element_type_default_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.float32);  mul_tensor_8 = None
        eq_tensor_2: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_4, convert_element_type_default_4)
        abs_default_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_4)
        ne_scalar_2: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_2, inf);  abs_default_2 = None
        mul_tensor_11: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_2, ne_scalar_2);  eq_tensor_2 = ne_scalar_2 = None
        logical_not_default_4: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_11);  mul_tensor_11 = None
        any_dims_2: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_4, [3], True);  logical_not_default_4 = None
        logical_not_default_5: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_2);  any_dims_2 = None
        convert_element_type_default_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.float32);  add_234 = None
        mul_tensor_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1);  convert_element_type_default_5 = None
        amax_default_4: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_9, [3], True)
        sub_tensor_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_9, amax_default_4);  mul_tensor_9 = amax_default_4 = None
        mul_tensor_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.125);  sub_tensor_4 = None
        amax_default_5: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_4, [3], True)
        sub_tensor_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_4, amax_default_5);  convert_element_type_default_4 = amax_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_5, mul_tensor_10, sub_tensor_5);  logical_not_default_5 = mul_tensor_10 = sub_tensor_5 = None
        exp_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_2);  where_self_2 = None
        sum_22: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [3], True)
        div_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_667: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_548: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_667, 4);  convert_element_type_667 = None
        view_824: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_548, [256, 512, 512]);  unsqueeze_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_536: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_652, 3)
        unsqueeze_537: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 4);  unsqueeze_536 = None
        view_806: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_537, [1, 8192, 1024]);  unsqueeze_537 = None
        squeeze_dim_24: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_806, 0);  view_806 = None
        unsqueeze_538: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg319_1, 3);  arg319_1 = None
        unsqueeze_539: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 4);  unsqueeze_538 = None
        view_807: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_539, [1, 1024, 1024]);  unsqueeze_539 = None
        squeeze_dim_25: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_807, 0);  view_807 = None
        mm_default_12: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_24, squeeze_dim_25);  squeeze_dim_24 = squeeze_dim_25 = None
        unsqueeze_default_12: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_12, 0);  mm_default_12 = None
        view_808: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_12, [512, 16, 1, 16, 64]);  unsqueeze_default_12 = None
        permute_899: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_808, [0, 1, 3, 4, 2]);  view_808 = None
        view_809: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_899, [512, 16, 16, 64]);  permute_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_549: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_809, 4);  view_809 = None
        permute_916: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_549, [4, 1, 2, 3, 0]);  unsqueeze_549 = None
        permute_918: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_916, [1, 2, 4, 3, 0]);  permute_916 = None
        view_825: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_918, [256, 512, 64]);  permute_918 = None
        bmm_174: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_824, view_825);  view_824 = view_825 = None
        view_826: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_174, [16, 16, 512, 1, 64]);  bmm_174 = None
        permute_919: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_826, [2, 0, 1, 4, 3]);  view_826 = None
        view_827: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_919, [512, 16, 16, 64]);  permute_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_550: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_827, 4);  view_827 = None
        permute_920: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_550, [0, 1, 4, 3, 2]);  unsqueeze_550 = None
        permute_922: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_920, [0, 1, 3, 4, 2]);  permute_920 = None
        clone_130: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_922, memory_format = torch.contiguous_format);  permute_922 = None
        view_828: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [1, 8192, 1024]);  clone_130 = None
        squeeze_dim_20: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_828, 0);  view_828 = None
        unsqueeze_551: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg323_1, 3);  arg323_1 = None
        unsqueeze_552: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 4);  unsqueeze_551 = None
        permute_921: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_552, [3, 4, 0, 2, 1]);  unsqueeze_552 = None
        permute_923: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_921, [3, 4, 2, 0, 1]);  permute_921 = None
        clone_131: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_923, memory_format = torch.contiguous_format);  permute_923 = None
        view_829: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [1, 1024, 1024]);  clone_131 = None
        squeeze_dim_21: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_829, 0);  view_829 = None
        mm_default_10: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_20, squeeze_dim_21);  squeeze_dim_20 = squeeze_dim_21 = None
        unsqueeze_default_10: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_10, 0);  mm_default_10 = None
        view_830: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_10, [512, 16, 1, 1, 1024]);  unsqueeze_default_10 = None
        permute_924: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_830, [0, 1, 4, 2, 3]);  view_830 = None
        view_831: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_924, [512, 16, 1024]);  permute_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_235: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_831, convert_element_type_652);  view_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_672: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_235, torch.float32);  add_235 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_672, [2], correction = 0, keepdim = True)
        getitem_84: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        sub_64: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_672, getitem_85);  convert_element_type_672 = getitem_85 = None
        add_236: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_171: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = rsqrt_42 = None
        mul_172: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, arg324_1);  mul_171 = arg324_1 = None
        add_237: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, arg325_1);  mul_172 = arg325_1 = None
        convert_element_type_673: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_832: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_673, [8192, 1024])
        permute_925: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_42: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg327_1, view_832, permute_925);  arg327_1 = view_832 = permute_925 = None
        view_833: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [512, 16, 4096]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_677: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_833, torch.float32);  view_833 = None
        mul_173: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_677, 0.5)
        mul_174: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_677, 0.7071067811865476);  convert_element_type_677 = None
        erf_21: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_238: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_175: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, add_238);  mul_173 = add_238 = None
        convert_element_type_678: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_834: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_678, [8192, 4096]);  convert_element_type_678 = None
        permute_926: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_43: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg329_1, view_834, permute_926);  arg329_1 = view_834 = permute_926 = None
        view_835: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [512, 16, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_239: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_835, convert_element_type_673);  view_835 = convert_element_type_673 = None
        convert_element_type_682: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_239, torch.float32);  add_239 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_682, [2], correction = 0, keepdim = True)
        getitem_86: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        sub_65: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_682, getitem_87);  convert_element_type_682 = getitem_87 = None
        add_240: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        mul_176: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
        mul_177: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, arg330_1);  mul_176 = arg330_1 = None
        add_241: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_177, arg331_1);  mul_177 = arg331_1 = None
        convert_element_type_683: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_241, torch.bfloat16);  add_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_553: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_683, 3)
        unsqueeze_554: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 4);  unsqueeze_553 = None
        view_836: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_554, [1, 8192, 1024]);  unsqueeze_554 = None
        squeeze_dim_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_836, 0);  view_836 = None
        unsqueeze_555: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg332_1, 3);  arg332_1 = None
        unsqueeze_556: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 4);  unsqueeze_555 = None
        view_837: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_556, [1, 1024, 1024]);  unsqueeze_556 = None
        squeeze_dim_19: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_837, 0);  view_837 = None
        mm_default_9: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_18, squeeze_dim_19);  squeeze_dim_18 = squeeze_dim_19 = None
        unsqueeze_default_9: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_9, 0);  mm_default_9 = None
        view_838: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_9, [512, 16, 1, 16, 64]);  unsqueeze_default_9 = None
        permute_931: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_838, [0, 1, 3, 4, 2]);  view_838 = None
        view_839: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_931, [512, 16, 16, 64]);  permute_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_242: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_839, arg336_1);  arg336_1 = None
        unsqueeze_569: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_242, 4);  add_242 = None
        permute_947: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_569, [1, 2, 0, 4, 3]);  unsqueeze_569 = None
        permute_949: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_947, [0, 1, 2, 4, 3]);  permute_947 = None
        view_852: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_949, [256, 512, 64]);  permute_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_557: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_683, 3)
        unsqueeze_558: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 4);  unsqueeze_557 = None
        view_840: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_558, [1, 8192, 1024]);  unsqueeze_558 = None
        squeeze_dim_16: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_840, 0);  view_840 = None
        unsqueeze_559: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg333_1, 3);  arg333_1 = None
        unsqueeze_560: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 4);  unsqueeze_559 = None
        view_841: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_560, [1, 1024, 1024]);  unsqueeze_560 = None
        squeeze_dim_17: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_841, 0);  view_841 = None
        mm_default_8: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_16, squeeze_dim_17);  squeeze_dim_16 = squeeze_dim_17 = None
        unsqueeze_default_8: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_8, 0);  mm_default_8 = None
        view_842: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_8, [512, 16, 1, 16, 64]);  unsqueeze_default_8 = None
        permute_936: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_842, [0, 1, 3, 4, 2]);  view_842 = None
        view_843: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_936, [512, 16, 16, 64]);  permute_936 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_570: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_843, 4);  view_843 = None
        permute_948: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_570, [1, 2, 4, 0, 3]);  unsqueeze_570 = None
        permute_950: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_948, [0, 1, 4, 3, 2]);  permute_948 = None
        view_853: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_950, [256, 64, 512]);  permute_950 = None
        bmm_180: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_852, view_853);  view_852 = view_853 = None
        view_854: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_180, [16, 16, 512, 1, 512]);  bmm_180 = None
        permute_951: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_854, [0, 1, 2, 4, 3]);  view_854 = None
        view_855: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_951, [16, 16, 512, 512]);  permute_951 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_243: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_839, arg337_1);  view_839 = arg337_1 = None
        unsqueeze_571: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_243, 4);  add_243 = None
        permute_952: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_571, [1, 2, 0, 4, 3]);  unsqueeze_571 = None
        permute_954: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_952, [0, 1, 2, 4, 3]);  permute_952 = None
        view_856: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_954, [256, 512, 64]);  permute_954 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_690: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16)
        unsqueeze_565: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_690, 3);  convert_element_type_690 = None
        unsqueeze_566: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 4);  unsqueeze_565 = None
        view_848: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_566, [1, 16384, 1024]);  unsqueeze_566 = None
        squeeze_dim_12: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_848, 0);  view_848 = None
        unsqueeze_567: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg335_1, 3);  arg335_1 = None
        unsqueeze_568: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 4);  unsqueeze_567 = None
        view_849: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_568, [1, 1024, 1024]);  unsqueeze_568 = None
        squeeze_dim_13: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_849, 0);  view_849 = None
        mm_default_6: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_12, squeeze_dim_13);  squeeze_dim_12 = squeeze_dim_13 = None
        unsqueeze_default_6: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_6, 0);  mm_default_6 = None
        view_850: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_6, [1024, 16, 1, 16, 64]);  unsqueeze_default_6 = None
        permute_946: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_850, [0, 1, 3, 4, 2]);  view_850 = None
        view_851: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_946, [1024, 16, 16, 64]);  permute_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_572: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_851, 4);  view_851 = None
        permute_953: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_572, [1, 2, 4, 0, 3]);  unsqueeze_572 = None
        permute_955: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_953, [0, 1, 4, 3, 2]);  permute_953 = None
        view_857: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_955, [256, 64, 1024]);  permute_955 = None
        bmm_181: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_856, view_857);  view_856 = view_857 = None
        view_858: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_181, [16, 16, 512, 1, 1024]);  bmm_181 = None
        permute_956: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_858, [0, 1, 2, 4, 3]);  view_858 = None
        view_859: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_956, [16, 16, 512, 1024]);  permute_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_860: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_859, [16, 16, 1024, 512]);  view_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_46: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_860, 2, 1, 9223372036854775807);  view_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_861: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_46, [16, 16, 512, 1023]);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_24: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_22: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_861, [None, None, None, iota_24]);  view_861 = iota_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_244: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_855, index_22);  view_855 = index_22 = None
        add_245: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_244, 0);  add_244 = None

        # No stacktrace found for following nodes
        mul_tensor_4: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, 0.125)
        convert_element_type_default_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float32);  mul_tensor_4 = None
        eq_tensor_1: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_2, convert_element_type_default_2)
        abs_default_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_2)
        ne_scalar_1: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_1, inf);  abs_default_1 = None
        mul_tensor_7: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_1, ne_scalar_1);  eq_tensor_1 = ne_scalar_1 = None
        logical_not_default_2: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_7);  mul_tensor_7 = None
        any_dims_1: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_2, [3], True);  logical_not_default_2 = None
        logical_not_default_3: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_1);  any_dims_1 = None
        convert_element_type_default_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_245, torch.float32);  add_245 = None
        mul_tensor_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1);  convert_element_type_default_3 = None
        amax_default_2: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_5, [3], True)
        sub_tensor_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_5, amax_default_2);  mul_tensor_5 = amax_default_2 = None
        mul_tensor_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.125);  sub_tensor_2 = None
        amax_default_3: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_2, [3], True)
        sub_tensor_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default_3);  convert_element_type_default_2 = amax_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_3, mul_tensor_6, sub_tensor_3);  logical_not_default_3 = mul_tensor_6 = sub_tensor_3 = None
        exp_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_1);  where_self_1 = None
        sum_23: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [3], True)
        div_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_698: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_573: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_698, 4);  convert_element_type_698 = None
        view_862: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_573, [256, 512, 512]);  unsqueeze_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_561: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_683, 3)
        unsqueeze_562: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 4);  unsqueeze_561 = None
        view_844: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_562, [1, 8192, 1024]);  unsqueeze_562 = None
        squeeze_dim_14: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_844, 0);  view_844 = None
        unsqueeze_563: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg334_1, 3);  arg334_1 = None
        unsqueeze_564: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 4);  unsqueeze_563 = None
        view_845: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_564, [1, 1024, 1024]);  unsqueeze_564 = None
        squeeze_dim_15: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_845, 0);  view_845 = None
        mm_default_7: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_14, squeeze_dim_15);  squeeze_dim_14 = squeeze_dim_15 = None
        unsqueeze_default_7: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_7, 0);  mm_default_7 = None
        view_846: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_7, [512, 16, 1, 16, 64]);  unsqueeze_default_7 = None
        permute_941: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_846, [0, 1, 3, 4, 2]);  view_846 = None
        view_847: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_941, [512, 16, 16, 64]);  permute_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_574: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_847, 4);  view_847 = None
        permute_958: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_574, [4, 1, 2, 3, 0]);  unsqueeze_574 = None
        permute_960: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_958, [1, 2, 4, 3, 0]);  permute_958 = None
        view_863: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_960, [256, 512, 64]);  permute_960 = None
        bmm_182: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_862, view_863);  view_862 = view_863 = None
        view_864: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_182, [16, 16, 512, 1, 64]);  bmm_182 = None
        permute_961: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_864, [2, 0, 1, 4, 3]);  view_864 = None
        view_865: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_961, [512, 16, 16, 64]);  permute_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_575: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_865, 4);  view_865 = None
        permute_962: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_575, [0, 1, 4, 3, 2]);  unsqueeze_575 = None
        permute_964: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_962, [0, 1, 3, 4, 2]);  permute_962 = None
        clone_136: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_964, memory_format = torch.contiguous_format);  permute_964 = None
        view_866: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [1, 8192, 1024]);  clone_136 = None
        squeeze_dim_10: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_866, 0);  view_866 = None
        unsqueeze_576: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg338_1, 3);  arg338_1 = None
        unsqueeze_577: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 4);  unsqueeze_576 = None
        permute_963: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_577, [3, 4, 0, 2, 1]);  unsqueeze_577 = None
        permute_965: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_963, [3, 4, 2, 0, 1]);  permute_963 = None
        clone_137: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_965, memory_format = torch.contiguous_format);  permute_965 = None
        view_867: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [1, 1024, 1024]);  clone_137 = None
        squeeze_dim_11: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_867, 0);  view_867 = None
        mm_default_5: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_10, squeeze_dim_11);  squeeze_dim_10 = squeeze_dim_11 = None
        unsqueeze_default_5: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_5, 0);  mm_default_5 = None
        view_868: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_5, [512, 16, 1, 1, 1024]);  unsqueeze_default_5 = None
        permute_966: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_868, [0, 1, 4, 2, 3]);  view_868 = None
        view_869: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_966, [512, 16, 1024]);  permute_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_246: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_869, convert_element_type_683);  view_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_703: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_246, torch.float32);  add_246 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_703, [2], correction = 0, keepdim = True)
        getitem_88: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        sub_67: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_703, getitem_89);  convert_element_type_703 = getitem_89 = None
        add_247: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_179: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = rsqrt_44 = None
        mul_180: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, arg339_1);  mul_179 = arg339_1 = None
        add_248: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_180, arg340_1);  mul_180 = arg340_1 = None
        convert_element_type_704: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_248, torch.bfloat16);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_870: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_704, [8192, 1024])
        permute_967: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_44: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg342_1, view_870, permute_967);  arg342_1 = view_870 = permute_967 = None
        view_871: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [512, 16, 4096]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_708: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_871, torch.float32);  view_871 = None
        mul_181: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, 0.5)
        mul_182: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, 0.7071067811865476);  convert_element_type_708 = None
        erf_22: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_249: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_183: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, add_249);  mul_181 = add_249 = None
        convert_element_type_709: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_183, torch.bfloat16);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_872: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_709, [8192, 4096]);  convert_element_type_709 = None
        permute_968: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_45: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg344_1, view_872, permute_968);  arg344_1 = view_872 = permute_968 = None
        view_873: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [512, 16, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_250: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_873, convert_element_type_704);  view_873 = convert_element_type_704 = None
        convert_element_type_713: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_250, torch.float32);  add_250 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_713, [2], correction = 0, keepdim = True)
        getitem_90: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        sub_68: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_713, getitem_91);  convert_element_type_713 = getitem_91 = None
        add_251: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        mul_184: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
        mul_185: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, arg345_1);  mul_184 = arg345_1 = None
        add_252: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_185, arg346_1);  mul_185 = arg346_1 = None
        convert_element_type_714: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16);  add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_578: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_714, 3)
        unsqueeze_579: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 4);  unsqueeze_578 = None
        view_874: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_579, [1, 8192, 1024]);  unsqueeze_579 = None
        squeeze_dim_8: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_874, 0);  view_874 = None
        unsqueeze_580: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg347_1, 3);  arg347_1 = None
        unsqueeze_581: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 4);  unsqueeze_580 = None
        view_875: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_581, [1, 1024, 1024]);  unsqueeze_581 = None
        squeeze_dim_9: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_875, 0);  view_875 = None
        mm_default_4: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_8, squeeze_dim_9);  squeeze_dim_8 = squeeze_dim_9 = None
        unsqueeze_default_4: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_4, 0);  mm_default_4 = None
        view_876: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_4, [512, 16, 1, 16, 64]);  unsqueeze_default_4 = None
        permute_973: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_876, [0, 1, 3, 4, 2]);  view_876 = None
        view_877: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_973, [512, 16, 16, 64]);  permute_973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_253: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_877, arg351_1);  arg351_1 = None
        unsqueeze_594: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_253, 4);  add_253 = None
        permute_989: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_594, [1, 2, 0, 4, 3]);  unsqueeze_594 = None
        permute_991: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_989, [0, 1, 2, 4, 3]);  permute_989 = None
        view_890: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_991, [256, 512, 64]);  permute_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_582: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_714, 3)
        unsqueeze_583: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 4);  unsqueeze_582 = None
        view_878: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_583, [1, 8192, 1024]);  unsqueeze_583 = None
        squeeze_dim_6: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_878, 0);  view_878 = None
        unsqueeze_584: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg348_1, 3);  arg348_1 = None
        unsqueeze_585: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 4);  unsqueeze_584 = None
        view_879: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_585, [1, 1024, 1024]);  unsqueeze_585 = None
        squeeze_dim_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_879, 0);  view_879 = None
        mm_default_3: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_6, squeeze_dim_7);  squeeze_dim_6 = squeeze_dim_7 = None
        unsqueeze_default_3: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_3, 0);  mm_default_3 = None
        view_880: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_3, [512, 16, 1, 16, 64]);  unsqueeze_default_3 = None
        permute_978: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_880, [0, 1, 3, 4, 2]);  view_880 = None
        view_881: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_978, [512, 16, 16, 64]);  permute_978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        unsqueeze_595: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_881, 4);  view_881 = None
        permute_990: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_595, [1, 2, 4, 0, 3]);  unsqueeze_595 = None
        permute_992: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_990, [0, 1, 4, 3, 2]);  permute_990 = None
        view_891: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_992, [256, 64, 512]);  permute_992 = None
        bmm_188: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_890, view_891);  view_890 = view_891 = None
        view_892: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_188, [16, 16, 512, 1, 512]);  bmm_188 = None
        permute_993: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_892, [0, 1, 2, 4, 3]);  view_892 = None
        view_893: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_993, [16, 16, 512, 512]);  permute_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_254: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_877, arg352_1);  view_877 = arg352_1 = None
        unsqueeze_596: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_254, 4);  add_254 = None
        permute_994: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_596, [1, 2, 0, 4, 3]);  unsqueeze_596 = None
        permute_996: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_994, [0, 1, 2, 4, 3]);  permute_994 = None
        view_894: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_996, [256, 512, 64]);  permute_996 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_721: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_2, torch.bfloat16);  clone_2 = None
        unsqueeze_590: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_721, 3);  convert_element_type_721 = None
        unsqueeze_591: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 4);  unsqueeze_590 = None
        view_886: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_591, [1, 16384, 1024]);  unsqueeze_591 = None
        squeeze_dim_2: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_886, 0);  view_886 = None
        unsqueeze_592: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg350_1, 3);  arg350_1 = None
        unsqueeze_593: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 4);  unsqueeze_592 = None
        view_887: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_593, [1, 1024, 1024]);  unsqueeze_593 = None
        squeeze_dim_3: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_887, 0);  view_887 = None
        mm_default_1: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_2, squeeze_dim_3);  squeeze_dim_2 = squeeze_dim_3 = None
        unsqueeze_default_1: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_1, 0);  mm_default_1 = None
        view_888: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_1, [1024, 16, 1, 16, 64]);  unsqueeze_default_1 = None
        permute_988: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_888, [0, 1, 3, 4, 2]);  view_888 = None
        view_889: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_988, [1024, 16, 16, 64]);  permute_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        unsqueeze_597: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_889, 4);  view_889 = None
        permute_995: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_597, [1, 2, 4, 0, 3]);  unsqueeze_597 = None
        permute_997: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_995, [0, 1, 4, 3, 2]);  permute_995 = None
        view_895: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_997, [256, 64, 1024]);  permute_997 = None
        bmm_189: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_894, view_895);  view_894 = view_895 = None
        view_896: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_189, [16, 16, 512, 1, 1024]);  bmm_189 = None
        permute_998: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_896, [0, 1, 2, 4, 3]);  view_896 = None
        view_897: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_998, [16, 16, 512, 1024]);  permute_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_898: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_897, [16, 16, 1024, 512]);  view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_48: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_898, 2, 1, 9223372036854775807);  view_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_899: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_48, [16, 16, 512, 1023]);  slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_25: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_23: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_899, [None, None, None, iota_25]);  view_899 = iota_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_255: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_893, index_23);  view_893 = index_23 = None
        add_256: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_255, 0);  add_255 = None

        # No stacktrace found for following nodes
        mul_tensor: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_256, 0.125)
        convert_element_type_default: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        eq_tensor: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default, convert_element_type_default)
        abs_default: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default)
        ne_scalar: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_3: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_3);  mul_tensor_3 = None
        any_dims: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default, [3], True);  logical_not_default = None
        logical_not_default_1: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None
        convert_element_type_default_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_256, torch.float32);  add_256 = None
        mul_tensor_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        amax_default: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_1, [3], True)
        sub_tensor: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = amax_default = None
        mul_tensor_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        amax_default_1: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default, [3], True)
        sub_tensor_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default_1);  convert_element_type_default = amax_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  logical_not_default_1 = mul_tensor_2 = sub_tensor_1 = None
        exp_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_24: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [3], True)
        div_24: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_729: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_598: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_729, 4);  convert_element_type_729 = None
        view_900: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_598, [256, 512, 512]);  unsqueeze_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_586: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_714, 3)
        unsqueeze_587: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 4);  unsqueeze_586 = None
        view_882: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_587, [1, 8192, 1024]);  unsqueeze_587 = None
        squeeze_dim_4: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_882, 0);  view_882 = None
        unsqueeze_588: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg349_1, 3);  arg349_1 = None
        unsqueeze_589: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 4);  unsqueeze_588 = None
        view_883: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_589, [1, 1024, 1024]);  unsqueeze_589 = None
        squeeze_dim_5: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_883, 0);  view_883 = None
        mm_default_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_4, squeeze_dim_5);  squeeze_dim_4 = squeeze_dim_5 = None
        unsqueeze_default_2: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_2, 0);  mm_default_2 = None
        view_884: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_2, [512, 16, 1, 16, 64]);  unsqueeze_default_2 = None
        permute_983: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_884, [0, 1, 3, 4, 2]);  view_884 = None
        view_885: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_983, [512, 16, 16, 64]);  permute_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        unsqueeze_599: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_885, 4);  view_885 = None
        permute_1000: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_599, [4, 1, 2, 3, 0]);  unsqueeze_599 = None
        permute_1002: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_1000, [1, 2, 4, 3, 0]);  permute_1000 = None
        view_901: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1002, [256, 512, 64]);  permute_1002 = None
        bmm_190: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_900, view_901);  view_900 = view_901 = None
        view_902: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_190, [16, 16, 512, 1, 64]);  bmm_190 = None
        permute_1003: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_902, [2, 0, 1, 4, 3]);  view_902 = None
        view_903: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1003, [512, 16, 16, 64]);  permute_1003 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_600: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_903, 4);  view_903 = None
        permute_1004: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_600, [0, 1, 4, 3, 2]);  unsqueeze_600 = None
        permute_1006: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1004, [0, 1, 3, 4, 2]);  permute_1004 = None
        clone_142: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_1006, memory_format = torch.contiguous_format);  permute_1006 = None
        view_904: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [1, 8192, 1024]);  clone_142 = None
        squeeze_dim: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_904, 0);  view_904 = None
        unsqueeze_601: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg353_1, 3);  arg353_1 = None
        unsqueeze_602: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 4);  unsqueeze_601 = None
        permute_1005: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_602, [3, 4, 0, 2, 1]);  unsqueeze_602 = None
        permute_1007: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_1005, [3, 4, 2, 0, 1]);  permute_1005 = None
        clone_143: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_1007, memory_format = torch.contiguous_format);  permute_1007 = None
        view_905: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_143, [1, 1024, 1024]);  clone_143 = None
        squeeze_dim_1: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_905, 0);  view_905 = None
        mm_default: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_1);  squeeze_dim = squeeze_dim_1 = None
        unsqueeze_default: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        view_906: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default, [512, 16, 1, 1, 1024]);  unsqueeze_default = None
        permute_1008: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_906, [0, 1, 4, 2, 3]);  view_906 = None
        view_907: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1008, [512, 16, 1024]);  permute_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_257: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_907, convert_element_type_714);  view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        convert_element_type_734: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_257, torch.float32);  add_257 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_734, [2], correction = 0, keepdim = True)
        getitem_92: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_46[0]
        getitem_93: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        sub_70: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_734, getitem_93);  convert_element_type_734 = getitem_93 = None
        add_258: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        mul_187: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = rsqrt_46 = None
        mul_188: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, arg354_1);  mul_187 = arg354_1 = None
        add_259: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, arg355_1);  mul_188 = arg355_1 = None
        convert_element_type_735: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        view_908: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_735, [8192, 1024])
        permute_1009: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
        addmm_46: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg357_1, view_908, permute_1009);  arg357_1 = view_908 = permute_1009 = None
        view_909: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [512, 16, 4096]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_739: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_909, torch.float32);  view_909 = None
        mul_189: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_739, 0.5)
        mul_190: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_739, 0.7071067811865476);  convert_element_type_739 = None
        erf_23: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_190);  mul_190 = None
        add_260: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_191: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, add_260);  mul_189 = add_260 = None
        convert_element_type_740: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_191, torch.bfloat16);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        view_910: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_740, [8192, 4096]);  convert_element_type_740 = None
        permute_1010: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
        addmm_47: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg359_1, view_910, permute_1010);  arg359_1 = view_910 = permute_1010 = None
        view_911: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [512, 16, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_261: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_911, convert_element_type_735);  view_911 = convert_element_type_735 = None
        convert_element_type_744: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_261, torch.float32);  add_261 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_744, [2], correction = 0, keepdim = True)
        getitem_94: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_47[0]
        getitem_95: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1432 in forward, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        view_915: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(arg363_1, [-1]);  arg363_1 = None
        ne_1: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_915, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        sub_71: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_744, getitem_95);  convert_element_type_744 = getitem_95 = None
        add_262: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
        mul_192: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
        mul_193: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, arg360_1);  mul_192 = arg360_1 = None
        add_263: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, arg361_1);  mul_193 = arg361_1 = None
        convert_element_type_745: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_263, torch.bfloat16);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1180 in forward, code: output = output.permute(1, 0, 2).contiguous()
        permute_1011: "bf16[16, 512, 1024][1024, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_745, [1, 0, 2]);  convert_element_type_745 = None
        clone_148: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(permute_1011, memory_format = torch.contiguous_format);  permute_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1426 in forward, code: logits = self.lm_loss(hidden_states[:, slice_indices, :])
        view_912: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_148, [8192, 1024]);  clone_148 = None
        permute_1012: "bf16[1024, 32000][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm_48: "bf16[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.addmm.default(arg362_1, view_912, permute_1012);  arg362_1 = view_912 = permute_1012 = None
        view_913: "bf16[16, 512, 32000][16384000, 32000, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [16, 512, 32000]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1432 in forward, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        view_914: "bf16[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.reshape.default(view_913, [-1, 32000])
        convert_element_type_749: "f32[8192, 32000][32000, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_914, torch.float32);  view_914 = None
        amax_24: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_749, [1], True)
        sub_72: "f32[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_749, amax_24);  convert_element_type_749 = amax_24 = None
        exp_24: "f32[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.exp.default(sub_72)
        sum_25: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_73: "f32[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_72, log);  sub_72 = log = None
        convert_element_type_750: "bf16[8192, 32000][32000, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_73, torch.bfloat16);  sub_73 = None
        ne: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_915, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne, view_915, full_default);  ne = full_default = None
        unsqueeze_603: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_750, 1, unsqueeze_603);  convert_element_type_750 = unsqueeze_603 = None
        squeeze: "bf16[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[8192][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        sum_27: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_915, -100);  view_915 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_751: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.bfloat16);  sum_26 = None
        div_25: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_751);  sum_27 = convert_element_type_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:923 in cache_mem, code: new_mem = curr_out[cutoff:]
        slice_1: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(embedding, 0, -512, 9223372036854775807);  embedding = None
        slice_3: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_32, 0, -512, 9223372036854775807);  convert_element_type_32 = None
        slice_5: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_63, 0, -512, 9223372036854775807);  convert_element_type_63 = None
        slice_7: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_94, 0, -512, 9223372036854775807);  convert_element_type_94 = None
        slice_9: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_125, 0, -512, 9223372036854775807);  convert_element_type_125 = None
        slice_11: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_156, 0, -512, 9223372036854775807);  convert_element_type_156 = None
        slice_13: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_187, 0, -512, 9223372036854775807);  convert_element_type_187 = None
        slice_15: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_218, 0, -512, 9223372036854775807);  convert_element_type_218 = None
        slice_17: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_249, 0, -512, 9223372036854775807);  convert_element_type_249 = None
        slice_19: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_280, 0, -512, 9223372036854775807);  convert_element_type_280 = None
        slice_21: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_311, 0, -512, 9223372036854775807);  convert_element_type_311 = None
        slice_23: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_342, 0, -512, 9223372036854775807);  convert_element_type_342 = None
        slice_25: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_373, 0, -512, 9223372036854775807);  convert_element_type_373 = None
        slice_27: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_404, 0, -512, 9223372036854775807);  convert_element_type_404 = None
        slice_29: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_435, 0, -512, 9223372036854775807);  convert_element_type_435 = None
        slice_31: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_466, 0, -512, 9223372036854775807);  convert_element_type_466 = None
        slice_33: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_497, 0, -512, 9223372036854775807);  convert_element_type_497 = None
        slice_35: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_528, 0, -512, 9223372036854775807);  convert_element_type_528 = None
        slice_37: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_559, 0, -512, 9223372036854775807);  convert_element_type_559 = None
        slice_39: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_590, 0, -512, 9223372036854775807);  convert_element_type_590 = None
        slice_41: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 0, -512, 9223372036854775807);  convert_element_type_621 = None
        slice_43: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_652, 0, -512, 9223372036854775807);  convert_element_type_652 = None
        slice_45: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_683, 0, -512, 9223372036854775807);  convert_element_type_683 = None
        slice_47: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_714, 0, -512, 9223372036854775807);  convert_element_type_714 = None
        return (div_25, view_913, slice_1, slice_3, slice_5, slice_7, slice_9, slice_11, slice_13, slice_15, slice_17, slice_19, slice_21, slice_23, slice_25, slice_27, slice_29, slice_31, slice_33, slice_35, slice_37, slice_39, slice_41, slice_43, slice_45, slice_47)

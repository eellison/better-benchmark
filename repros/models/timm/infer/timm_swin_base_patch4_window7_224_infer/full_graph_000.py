class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", arg1_1: "f32[128, 3, 4, 4][48, 1, 12, 3]cuda:0", arg2_1: "f32[128][1]cuda:0", arg3_1: "f32[128][1]cuda:0", arg4_1: "f32[128][1]cuda:0", arg5_1: "f32[128][1]cuda:0", arg6_1: "f32[128][1]cuda:0", arg7_1: "f32[384, 128][128, 1]cuda:0", arg8_1: "f32[384][1]cuda:0", arg9_1: "f32[169, 4][4, 1]cuda:0", arg10_1: "i64[49, 49][49, 1]cuda:0", arg11_1: "f32[128, 128][128, 1]cuda:0", arg12_1: "f32[128][1]cuda:0", arg13_1: "f32[128][1]cuda:0", arg14_1: "f32[128][1]cuda:0", arg15_1: "f32[512, 128][128, 1]cuda:0", arg16_1: "f32[512][1]cuda:0", arg17_1: "f32[128, 512][512, 1]cuda:0", arg18_1: "f32[128][1]cuda:0", arg19_1: "f32[128][1]cuda:0", arg20_1: "f32[128][1]cuda:0", arg21_1: "f32[64, 49, 49][2401, 49, 1]cuda:0", arg22_1: "f32[384, 128][128, 1]cuda:0", arg23_1: "f32[384][1]cuda:0", arg24_1: "f32[169, 4][4, 1]cuda:0", arg25_1: "i64[49, 49][49, 1]cuda:0", arg26_1: "f32[128, 128][128, 1]cuda:0", arg27_1: "f32[128][1]cuda:0", arg28_1: "f32[128][1]cuda:0", arg29_1: "f32[128][1]cuda:0", arg30_1: "f32[512, 128][128, 1]cuda:0", arg31_1: "f32[512][1]cuda:0", arg32_1: "f32[128, 512][512, 1]cuda:0", arg33_1: "f32[128][1]cuda:0", arg34_1: "f32[512][1]cuda:0", arg35_1: "f32[512][1]cuda:0", arg36_1: "f32[256, 512][512, 1]cuda:0", arg37_1: "f32[256][1]cuda:0", arg38_1: "f32[256][1]cuda:0", arg39_1: "f32[768, 256][256, 1]cuda:0", arg40_1: "f32[768][1]cuda:0", arg41_1: "f32[169, 8][8, 1]cuda:0", arg42_1: "i64[49, 49][49, 1]cuda:0", arg43_1: "f32[256, 256][256, 1]cuda:0", arg44_1: "f32[256][1]cuda:0", arg45_1: "f32[256][1]cuda:0", arg46_1: "f32[256][1]cuda:0", arg47_1: "f32[1024, 256][256, 1]cuda:0", arg48_1: "f32[1024][1]cuda:0", arg49_1: "f32[256, 1024][1024, 1]cuda:0", arg50_1: "f32[256][1]cuda:0", arg51_1: "f32[256][1]cuda:0", arg52_1: "f32[256][1]cuda:0", arg53_1: "f32[16, 49, 49][2401, 49, 1]cuda:0", arg54_1: "f32[768, 256][256, 1]cuda:0", arg55_1: "f32[768][1]cuda:0", arg56_1: "f32[169, 8][8, 1]cuda:0", arg57_1: "i64[49, 49][49, 1]cuda:0", arg58_1: "f32[256, 256][256, 1]cuda:0", arg59_1: "f32[256][1]cuda:0", arg60_1: "f32[256][1]cuda:0", arg61_1: "f32[256][1]cuda:0", arg62_1: "f32[1024, 256][256, 1]cuda:0", arg63_1: "f32[1024][1]cuda:0", arg64_1: "f32[256, 1024][1024, 1]cuda:0", arg65_1: "f32[256][1]cuda:0", arg66_1: "f32[1024][1]cuda:0", arg67_1: "f32[1024][1]cuda:0", arg68_1: "f32[512, 1024][1024, 1]cuda:0", arg69_1: "f32[512][1]cuda:0", arg70_1: "f32[512][1]cuda:0", arg71_1: "f32[1536, 512][512, 1]cuda:0", arg72_1: "f32[1536][1]cuda:0", arg73_1: "f32[169, 16][16, 1]cuda:0", arg74_1: "i64[49, 49][49, 1]cuda:0", arg75_1: "f32[512, 512][512, 1]cuda:0", arg76_1: "f32[512][1]cuda:0", arg77_1: "f32[512][1]cuda:0", arg78_1: "f32[512][1]cuda:0", arg79_1: "f32[2048, 512][512, 1]cuda:0", arg80_1: "f32[2048][1]cuda:0", arg81_1: "f32[512, 2048][2048, 1]cuda:0", arg82_1: "f32[512][1]cuda:0", arg83_1: "f32[512][1]cuda:0", arg84_1: "f32[512][1]cuda:0", arg85_1: "f32[4, 49, 49][2401, 49, 1]cuda:0", arg86_1: "f32[1536, 512][512, 1]cuda:0", arg87_1: "f32[1536][1]cuda:0", arg88_1: "f32[169, 16][16, 1]cuda:0", arg89_1: "i64[49, 49][49, 1]cuda:0", arg90_1: "f32[512, 512][512, 1]cuda:0", arg91_1: "f32[512][1]cuda:0", arg92_1: "f32[512][1]cuda:0", arg93_1: "f32[512][1]cuda:0", arg94_1: "f32[2048, 512][512, 1]cuda:0", arg95_1: "f32[2048][1]cuda:0", arg96_1: "f32[512, 2048][2048, 1]cuda:0", arg97_1: "f32[512][1]cuda:0", arg98_1: "f32[512][1]cuda:0", arg99_1: "f32[512][1]cuda:0", arg100_1: "f32[1536, 512][512, 1]cuda:0", arg101_1: "f32[1536][1]cuda:0", arg102_1: "f32[169, 16][16, 1]cuda:0", arg103_1: "i64[49, 49][49, 1]cuda:0", arg104_1: "f32[512, 512][512, 1]cuda:0", arg105_1: "f32[512][1]cuda:0", arg106_1: "f32[512][1]cuda:0", arg107_1: "f32[512][1]cuda:0", arg108_1: "f32[2048, 512][512, 1]cuda:0", arg109_1: "f32[2048][1]cuda:0", arg110_1: "f32[512, 2048][2048, 1]cuda:0", arg111_1: "f32[512][1]cuda:0", arg112_1: "f32[512][1]cuda:0", arg113_1: "f32[512][1]cuda:0", arg114_1: "f32[4, 49, 49][2401, 49, 1]cuda:0", arg115_1: "f32[1536, 512][512, 1]cuda:0", arg116_1: "f32[1536][1]cuda:0", arg117_1: "f32[169, 16][16, 1]cuda:0", arg118_1: "i64[49, 49][49, 1]cuda:0", arg119_1: "f32[512, 512][512, 1]cuda:0", arg120_1: "f32[512][1]cuda:0", arg121_1: "f32[512][1]cuda:0", arg122_1: "f32[512][1]cuda:0", arg123_1: "f32[2048, 512][512, 1]cuda:0", arg124_1: "f32[2048][1]cuda:0", arg125_1: "f32[512, 2048][2048, 1]cuda:0", arg126_1: "f32[512][1]cuda:0", arg127_1: "f32[512][1]cuda:0", arg128_1: "f32[512][1]cuda:0", arg129_1: "f32[1536, 512][512, 1]cuda:0", arg130_1: "f32[1536][1]cuda:0", arg131_1: "f32[169, 16][16, 1]cuda:0", arg132_1: "i64[49, 49][49, 1]cuda:0", arg133_1: "f32[512, 512][512, 1]cuda:0", arg134_1: "f32[512][1]cuda:0", arg135_1: "f32[512][1]cuda:0", arg136_1: "f32[512][1]cuda:0", arg137_1: "f32[2048, 512][512, 1]cuda:0", arg138_1: "f32[2048][1]cuda:0", arg139_1: "f32[512, 2048][2048, 1]cuda:0", arg140_1: "f32[512][1]cuda:0", arg141_1: "f32[512][1]cuda:0", arg142_1: "f32[512][1]cuda:0", arg143_1: "f32[4, 49, 49][2401, 49, 1]cuda:0", arg144_1: "f32[1536, 512][512, 1]cuda:0", arg145_1: "f32[1536][1]cuda:0", arg146_1: "f32[169, 16][16, 1]cuda:0", arg147_1: "i64[49, 49][49, 1]cuda:0", arg148_1: "f32[512, 512][512, 1]cuda:0", arg149_1: "f32[512][1]cuda:0", arg150_1: "f32[512][1]cuda:0", arg151_1: "f32[512][1]cuda:0", arg152_1: "f32[2048, 512][512, 1]cuda:0", arg153_1: "f32[2048][1]cuda:0", arg154_1: "f32[512, 2048][2048, 1]cuda:0", arg155_1: "f32[512][1]cuda:0", arg156_1: "f32[512][1]cuda:0", arg157_1: "f32[512][1]cuda:0", arg158_1: "f32[1536, 512][512, 1]cuda:0", arg159_1: "f32[1536][1]cuda:0", arg160_1: "f32[169, 16][16, 1]cuda:0", arg161_1: "i64[49, 49][49, 1]cuda:0", arg162_1: "f32[512, 512][512, 1]cuda:0", arg163_1: "f32[512][1]cuda:0", arg164_1: "f32[512][1]cuda:0", arg165_1: "f32[512][1]cuda:0", arg166_1: "f32[2048, 512][512, 1]cuda:0", arg167_1: "f32[2048][1]cuda:0", arg168_1: "f32[512, 2048][2048, 1]cuda:0", arg169_1: "f32[512][1]cuda:0", arg170_1: "f32[512][1]cuda:0", arg171_1: "f32[512][1]cuda:0", arg172_1: "f32[4, 49, 49][2401, 49, 1]cuda:0", arg173_1: "f32[1536, 512][512, 1]cuda:0", arg174_1: "f32[1536][1]cuda:0", arg175_1: "f32[169, 16][16, 1]cuda:0", arg176_1: "i64[49, 49][49, 1]cuda:0", arg177_1: "f32[512, 512][512, 1]cuda:0", arg178_1: "f32[512][1]cuda:0", arg179_1: "f32[512][1]cuda:0", arg180_1: "f32[512][1]cuda:0", arg181_1: "f32[2048, 512][512, 1]cuda:0", arg182_1: "f32[2048][1]cuda:0", arg183_1: "f32[512, 2048][2048, 1]cuda:0", arg184_1: "f32[512][1]cuda:0", arg185_1: "f32[512][1]cuda:0", arg186_1: "f32[512][1]cuda:0", arg187_1: "f32[1536, 512][512, 1]cuda:0", arg188_1: "f32[1536][1]cuda:0", arg189_1: "f32[169, 16][16, 1]cuda:0", arg190_1: "i64[49, 49][49, 1]cuda:0", arg191_1: "f32[512, 512][512, 1]cuda:0", arg192_1: "f32[512][1]cuda:0", arg193_1: "f32[512][1]cuda:0", arg194_1: "f32[512][1]cuda:0", arg195_1: "f32[2048, 512][512, 1]cuda:0", arg196_1: "f32[2048][1]cuda:0", arg197_1: "f32[512, 2048][2048, 1]cuda:0", arg198_1: "f32[512][1]cuda:0", arg199_1: "f32[512][1]cuda:0", arg200_1: "f32[512][1]cuda:0", arg201_1: "f32[4, 49, 49][2401, 49, 1]cuda:0", arg202_1: "f32[1536, 512][512, 1]cuda:0", arg203_1: "f32[1536][1]cuda:0", arg204_1: "f32[169, 16][16, 1]cuda:0", arg205_1: "i64[49, 49][49, 1]cuda:0", arg206_1: "f32[512, 512][512, 1]cuda:0", arg207_1: "f32[512][1]cuda:0", arg208_1: "f32[512][1]cuda:0", arg209_1: "f32[512][1]cuda:0", arg210_1: "f32[2048, 512][512, 1]cuda:0", arg211_1: "f32[2048][1]cuda:0", arg212_1: "f32[512, 2048][2048, 1]cuda:0", arg213_1: "f32[512][1]cuda:0", arg214_1: "f32[512][1]cuda:0", arg215_1: "f32[512][1]cuda:0", arg216_1: "f32[1536, 512][512, 1]cuda:0", arg217_1: "f32[1536][1]cuda:0", arg218_1: "f32[169, 16][16, 1]cuda:0", arg219_1: "i64[49, 49][49, 1]cuda:0", arg220_1: "f32[512, 512][512, 1]cuda:0", arg221_1: "f32[512][1]cuda:0", arg222_1: "f32[512][1]cuda:0", arg223_1: "f32[512][1]cuda:0", arg224_1: "f32[2048, 512][512, 1]cuda:0", arg225_1: "f32[2048][1]cuda:0", arg226_1: "f32[512, 2048][2048, 1]cuda:0", arg227_1: "f32[512][1]cuda:0", arg228_1: "f32[512][1]cuda:0", arg229_1: "f32[512][1]cuda:0", arg230_1: "f32[4, 49, 49][2401, 49, 1]cuda:0", arg231_1: "f32[1536, 512][512, 1]cuda:0", arg232_1: "f32[1536][1]cuda:0", arg233_1: "f32[169, 16][16, 1]cuda:0", arg234_1: "i64[49, 49][49, 1]cuda:0", arg235_1: "f32[512, 512][512, 1]cuda:0", arg236_1: "f32[512][1]cuda:0", arg237_1: "f32[512][1]cuda:0", arg238_1: "f32[512][1]cuda:0", arg239_1: "f32[2048, 512][512, 1]cuda:0", arg240_1: "f32[2048][1]cuda:0", arg241_1: "f32[512, 2048][2048, 1]cuda:0", arg242_1: "f32[512][1]cuda:0", arg243_1: "f32[512][1]cuda:0", arg244_1: "f32[512][1]cuda:0", arg245_1: "f32[1536, 512][512, 1]cuda:0", arg246_1: "f32[1536][1]cuda:0", arg247_1: "f32[169, 16][16, 1]cuda:0", arg248_1: "i64[49, 49][49, 1]cuda:0", arg249_1: "f32[512, 512][512, 1]cuda:0", arg250_1: "f32[512][1]cuda:0", arg251_1: "f32[512][1]cuda:0", arg252_1: "f32[512][1]cuda:0", arg253_1: "f32[2048, 512][512, 1]cuda:0", arg254_1: "f32[2048][1]cuda:0", arg255_1: "f32[512, 2048][2048, 1]cuda:0", arg256_1: "f32[512][1]cuda:0", arg257_1: "f32[512][1]cuda:0", arg258_1: "f32[512][1]cuda:0", arg259_1: "f32[4, 49, 49][2401, 49, 1]cuda:0", arg260_1: "f32[1536, 512][512, 1]cuda:0", arg261_1: "f32[1536][1]cuda:0", arg262_1: "f32[169, 16][16, 1]cuda:0", arg263_1: "i64[49, 49][49, 1]cuda:0", arg264_1: "f32[512, 512][512, 1]cuda:0", arg265_1: "f32[512][1]cuda:0", arg266_1: "f32[512][1]cuda:0", arg267_1: "f32[512][1]cuda:0", arg268_1: "f32[2048, 512][512, 1]cuda:0", arg269_1: "f32[2048][1]cuda:0", arg270_1: "f32[512, 2048][2048, 1]cuda:0", arg271_1: "f32[512][1]cuda:0", arg272_1: "f32[512][1]cuda:0", arg273_1: "f32[512][1]cuda:0", arg274_1: "f32[1536, 512][512, 1]cuda:0", arg275_1: "f32[1536][1]cuda:0", arg276_1: "f32[169, 16][16, 1]cuda:0", arg277_1: "i64[49, 49][49, 1]cuda:0", arg278_1: "f32[512, 512][512, 1]cuda:0", arg279_1: "f32[512][1]cuda:0", arg280_1: "f32[512][1]cuda:0", arg281_1: "f32[512][1]cuda:0", arg282_1: "f32[2048, 512][512, 1]cuda:0", arg283_1: "f32[2048][1]cuda:0", arg284_1: "f32[512, 2048][2048, 1]cuda:0", arg285_1: "f32[512][1]cuda:0", arg286_1: "f32[512][1]cuda:0", arg287_1: "f32[512][1]cuda:0", arg288_1: "f32[4, 49, 49][2401, 49, 1]cuda:0", arg289_1: "f32[1536, 512][512, 1]cuda:0", arg290_1: "f32[1536][1]cuda:0", arg291_1: "f32[169, 16][16, 1]cuda:0", arg292_1: "i64[49, 49][49, 1]cuda:0", arg293_1: "f32[512, 512][512, 1]cuda:0", arg294_1: "f32[512][1]cuda:0", arg295_1: "f32[512][1]cuda:0", arg296_1: "f32[512][1]cuda:0", arg297_1: "f32[2048, 512][512, 1]cuda:0", arg298_1: "f32[2048][1]cuda:0", arg299_1: "f32[512, 2048][2048, 1]cuda:0", arg300_1: "f32[512][1]cuda:0", arg301_1: "f32[512][1]cuda:0", arg302_1: "f32[512][1]cuda:0", arg303_1: "f32[1536, 512][512, 1]cuda:0", arg304_1: "f32[1536][1]cuda:0", arg305_1: "f32[169, 16][16, 1]cuda:0", arg306_1: "i64[49, 49][49, 1]cuda:0", arg307_1: "f32[512, 512][512, 1]cuda:0", arg308_1: "f32[512][1]cuda:0", arg309_1: "f32[512][1]cuda:0", arg310_1: "f32[512][1]cuda:0", arg311_1: "f32[2048, 512][512, 1]cuda:0", arg312_1: "f32[2048][1]cuda:0", arg313_1: "f32[512, 2048][2048, 1]cuda:0", arg314_1: "f32[512][1]cuda:0", arg315_1: "f32[512][1]cuda:0", arg316_1: "f32[512][1]cuda:0", arg317_1: "f32[4, 49, 49][2401, 49, 1]cuda:0", arg318_1: "f32[1536, 512][512, 1]cuda:0", arg319_1: "f32[1536][1]cuda:0", arg320_1: "f32[169, 16][16, 1]cuda:0", arg321_1: "i64[49, 49][49, 1]cuda:0", arg322_1: "f32[512, 512][512, 1]cuda:0", arg323_1: "f32[512][1]cuda:0", arg324_1: "f32[512][1]cuda:0", arg325_1: "f32[512][1]cuda:0", arg326_1: "f32[2048, 512][512, 1]cuda:0", arg327_1: "f32[2048][1]cuda:0", arg328_1: "f32[512, 2048][2048, 1]cuda:0", arg329_1: "f32[512][1]cuda:0", arg330_1: "f32[2048][1]cuda:0", arg331_1: "f32[2048][1]cuda:0", arg332_1: "f32[1024, 2048][2048, 1]cuda:0", arg333_1: "f32[1024][1]cuda:0", arg334_1: "f32[1024][1]cuda:0", arg335_1: "f32[3072, 1024][1024, 1]cuda:0", arg336_1: "f32[3072][1]cuda:0", arg337_1: "f32[169, 32][32, 1]cuda:0", arg338_1: "i64[49, 49][49, 1]cuda:0", arg339_1: "f32[1024, 1024][1024, 1]cuda:0", arg340_1: "f32[1024][1]cuda:0", arg341_1: "f32[1024][1]cuda:0", arg342_1: "f32[1024][1]cuda:0", arg343_1: "f32[4096, 1024][1024, 1]cuda:0", arg344_1: "f32[4096][1]cuda:0", arg345_1: "f32[1024, 4096][4096, 1]cuda:0", arg346_1: "f32[1024][1]cuda:0", arg347_1: "f32[1024][1]cuda:0", arg348_1: "f32[1024][1]cuda:0", arg349_1: "f32[3072, 1024][1024, 1]cuda:0", arg350_1: "f32[3072][1]cuda:0", arg351_1: "f32[169, 32][32, 1]cuda:0", arg352_1: "i64[49, 49][49, 1]cuda:0", arg353_1: "f32[1024, 1024][1024, 1]cuda:0", arg354_1: "f32[1024][1]cuda:0", arg355_1: "f32[1024][1]cuda:0", arg356_1: "f32[1024][1]cuda:0", arg357_1: "f32[4096, 1024][1024, 1]cuda:0", arg358_1: "f32[4096][1]cuda:0", arg359_1: "f32[1024, 4096][4096, 1]cuda:0", arg360_1: "f32[1024][1]cuda:0", arg361_1: "f32[1024][1]cuda:0", arg362_1: "f32[1024][1]cuda:0", arg363_1: "f32[1000, 1024][1024, 1]cuda:0", arg364_1: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convolution: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/format.py:68 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        var_mean = torch.ops.aten.var_mean.correction(permute, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute, getitem_1);  permute = getitem_1 = None
        add: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_1 = torch.ops.aten.var_mean.correction(add_1, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_1, getitem_3);  getitem_3 = None
        add_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        add_3: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, arg6_1);  mul_3 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_3, [128, 8, 7, 8, 7, 128]);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_1: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 896, 7168, 128, 1]cuda:0" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
        clone: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_1: "f32[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [-1, 7, 7, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_2: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [-1, 49, 128]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_3: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_2, [401408, 128]);  view_2 = None
        permute_2: "f32[128, 384][1, 128]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm: "f32[401408, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg8_1, view_3, permute_2);  arg8_1 = view_3 = permute_2 = None
        view_4: "f32[8192, 49, 384][18816, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [8192, 49, 384]);  addmm = None
        view_5: "f32[8192, 49, 3, 4, 32][18816, 384, 128, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [8192, 49, 3, 4, -1]);  view_4 = None
        permute_3: "f32[3, 8192, 4, 49, 32][128, 18816, 32, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_4: "f32[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind[0]
        getitem_5: "f32[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind[1]
        getitem_6: "f32[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_4: "f32[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_4, 0.1767766952966369);  getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand: "f32[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_4, [8192, 4, 49, 32]);  mul_4 = None
        clone_1: "f32[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_6: "f32[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [32768, 49, 32]);  clone_1 = None
        permute_4: "f32[8192, 4, 32, 49][18816, 32, 1, 384]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 1, 3, 2]);  getitem_5 = None
        expand_1: "f32[8192, 4, 32, 49][18816, 32, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_4, [8192, 4, 32, 49]);  permute_4 = None
        clone_2: "f32[8192, 4, 32, 49][6272, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "f32[32768, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [32768, 32, 49]);  clone_2 = None
        bmm: "f32[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
        view_8: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [8192, 4, 49, 49]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_9: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg10_1, [-1]);  arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index: "f32[2401, 4][4, 1]cuda:0" = torch.ops.aten.index.Tensor(arg9_1, [view_9]);  arg9_1 = view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_10: "f32[49, 49, 4][196, 4, 1]cuda:0" = torch.ops.aten.reshape.default(index, [49, 49, -1]);  index = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_5: "f32[4, 49, 49][1, 196, 4]cuda:0" = torch.ops.aten.permute.default(view_10, [2, 0, 1]);  view_10 = None
        clone_3: "f32[4, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze: "f32[1, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_3, 0);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_4: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_8, unsqueeze);  view_8 = unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_4, [-1], True)
        sub_2: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
        exp: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_2: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div, [8192, 4, 49, 49]);  div = None
        view_11: "f32[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_2, [32768, 49, 49]);  expand_2 = None
        expand_3: "f32[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = torch.ops.aten.expand.default(getitem_6, [8192, 4, 49, 32]);  getitem_6 = None
        clone_5: "f32[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12: "f32[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [32768, 49, 32]);  clone_5 = None
        bmm_1: "f32[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_11, view_12);  view_11 = view_12 = None
        view_13: "f32[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [8192, 4, 49, 32]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_6: "f32[8192, 49, 4, 32][6272, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        clone_6: "f32[8192, 49, 4, 32][6272, 128, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_14: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [8192, 49, 128]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_15: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [401408, 128]);  view_14 = None
        permute_7: "f32[128, 128][1, 128]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view_15, permute_7);  arg12_1 = view_15 = permute_7 = None
        view_16: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [8192, 49, 128]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_17: "f32[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_16, [-1, 7, 7, 128]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_18: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_17, [-1, 8, 8, 7, 7, 128]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_8: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 896, 6272, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 1, 3, 2, 4, 5]);  view_18 = None
        clone_8: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_19: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [-1, 56, 56, 128]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_5: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1, view_19);  add_1 = view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_20: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_5, [128, -1, 128]);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_2 = torch.ops.aten.var_mean.correction(view_20, [2], correction = 0, keepdim = True)
        getitem_7: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = var_mean_2[0]
        getitem_8: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_20, getitem_8);  getitem_8 = None
        add_6: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_7, 1e-05);  getitem_7 = None
        rsqrt_2: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_5: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_6: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, arg13_1);  mul_5 = arg13_1 = None
        add_7: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, arg14_1);  mul_6 = arg14_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_21: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(add_7, [401408, 128]);  add_7 = None
        permute_9: "f32[128, 512][1, 128]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_2: "f32[401408, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, view_21, permute_9);  arg16_1 = view_21 = permute_9 = None
        view_22: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 3136, 512]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_7: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        mul_8: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476);  view_22 = None
        erf: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_8: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, add_8);  mul_7 = add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_23: "f32[401408, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_9, [401408, 512]);  mul_9 = None
        permute_10: "f32[512, 128][1, 512]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_3: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.addmm.default(arg18_1, view_23, permute_10);  arg18_1 = view_23 = permute_10 = None
        view_24: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [128, 3136, 128]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_9: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_20, view_24);  view_20 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_25: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_9, [128, 56, 56, 128]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_3 = torch.ops.aten.var_mean.correction(view_25, [3], correction = 0, keepdim = True)
        getitem_9: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_3[0]
        getitem_10: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_4: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_25, getitem_10);  getitem_10 = None
        add_10: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_3: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_10: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_11: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, arg19_1);  mul_10 = arg19_1 = None
        add_11: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, arg20_1);  mul_11 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota: "i64[56][1]cuda:0" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_12: "i64[56][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 3);  iota = None
        fmod: "i64[56][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_12, 56);  add_12 = None
        index_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(add_11, [None, fmod]);  add_11 = fmod = None
        iota_1: "i64[56][1]cuda:0" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_13: "i64[56][1]cuda:0" = torch.ops.aten.add.Tensor(iota_1, 3);  iota_1 = None
        fmod_1: "i64[56][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_13, 56);  add_13 = None
        index_2: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(index_1, [None, None, fmod_1]);  index_1 = fmod_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_26: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(index_2, [128, 8, 7, 8, 7, 128]);  index_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_11: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 896, 7168, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 1, 3, 2, 4, 5]);  view_26 = None
        clone_11: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_27: "f32[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [-1, 7, 7, 128]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_28: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_27, [-1, 49, 128]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_29: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_28, [401408, 128]);  view_28 = None
        permute_12: "f32[128, 384][1, 128]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_4: "f32[401408, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg23_1, view_29, permute_12);  arg23_1 = view_29 = permute_12 = None
        view_30: "f32[8192, 49, 384][18816, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [8192, 49, 384]);  addmm_4 = None
        view_31: "f32[8192, 49, 3, 4, 32][18816, 384, 128, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [8192, 49, 3, 4, -1]);  view_30 = None
        permute_13: "f32[3, 8192, 4, 49, 32][128, 18816, 32, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_13);  permute_13 = None
        getitem_11: "f32[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind_1[0]
        getitem_12: "f32[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind_1[1]
        getitem_13: "f32[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_12: "f32[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_11, 0.1767766952966369);  getitem_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_4: "f32[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_12, [8192, 4, 49, 32]);  mul_12 = None
        clone_12: "f32[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_32: "f32[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [32768, 49, 32]);  clone_12 = None
        permute_14: "f32[8192, 4, 32, 49][18816, 32, 1, 384]cuda:0" = torch.ops.aten.permute.default(getitem_12, [0, 1, 3, 2]);  getitem_12 = None
        expand_5: "f32[8192, 4, 32, 49][18816, 32, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_14, [8192, 4, 32, 49]);  permute_14 = None
        clone_13: "f32[8192, 4, 32, 49][6272, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_33: "f32[32768, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [32768, 32, 49]);  clone_13 = None
        bmm_2: "f32[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_32, view_33);  view_32 = view_33 = None
        view_34: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [8192, 4, 49, 49]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_35: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg25_1, [-1]);  arg25_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_3: "f32[2401, 4][4, 1]cuda:0" = torch.ops.aten.index.Tensor(arg24_1, [view_35]);  arg24_1 = view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_36: "f32[49, 49, 4][196, 4, 1]cuda:0" = torch.ops.aten.reshape.default(index_3, [49, 49, -1]);  index_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_15: "f32[4, 49, 49][1, 196, 4]cuda:0" = torch.ops.aten.permute.default(view_36, [2, 0, 1]);  view_36 = None
        clone_14: "f32[4, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_1: "f32[1, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_14, 0);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_14: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_34, unsqueeze_1);  view_34 = unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_37: "f32[128, 64, 4, 49, 49][614656, 9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_14, [-1, 64, 4, 49, 49]);  add_14 = None
        unsqueeze_2: "f32[64, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg21_1, 1);  arg21_1 = None
        unsqueeze_3: "f32[1, 64, 1, 49, 49][153664, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 0);  unsqueeze_2 = None
        add_15: "f32[128, 64, 4, 49, 49][614656, 9604, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_37, unsqueeze_3);  view_37 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_38: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_15, [-1, 4, 49, 49]);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_1: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_38, [-1], True)
        sub_5: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_38, amax_1);  view_38 = amax_1 = None
        exp_1: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_6: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_1, [8192, 4, 49, 49]);  div_1 = None
        view_39: "f32[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_6, [32768, 49, 49]);  expand_6 = None
        expand_7: "f32[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = torch.ops.aten.expand.default(getitem_13, [8192, 4, 49, 32]);  getitem_13 = None
        clone_16: "f32[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_40: "f32[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [32768, 49, 32]);  clone_16 = None
        bmm_3: "f32[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_39, view_40);  view_39 = view_40 = None
        view_41: "f32[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [8192, 4, 49, 32]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_16: "f32[8192, 49, 4, 32][6272, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_17: "f32[8192, 49, 4, 32][6272, 128, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_42: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [8192, 49, 128]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_43: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [401408, 128]);  view_42 = None
        permute_17: "f32[128, 128][1, 128]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_5: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.addmm.default(arg27_1, view_43, permute_17);  arg27_1 = view_43 = permute_17 = None
        view_44: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [8192, 49, 128]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_45: "f32[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [-1, 7, 7, 128]);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_46: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [-1, 8, 8, 7, 7, 128]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_18: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 896, 6272, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 2, 4, 5]);  view_46 = None
        clone_19: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_47: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [-1, 56, 56, 128]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_2: "i64[56][1]cuda:0" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_16: "i64[56][1]cuda:0" = torch.ops.aten.add.Tensor(iota_2, 53);  iota_2 = None
        fmod_2: "i64[56][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_16, 56);  add_16 = None
        index_4: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(view_47, [None, fmod_2]);  view_47 = fmod_2 = None
        iota_3: "i64[56][1]cuda:0" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_17: "i64[56][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 53);  iota_3 = None
        fmod_3: "i64[56][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_17, 56);  add_17 = None
        index_5: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(index_4, [None, None, fmod_3]);  index_4 = fmod_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_18: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_25, index_5);  view_25 = index_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_48: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_18, [128, -1, 128]);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_4 = torch.ops.aten.var_mean.correction(view_48, [2], correction = 0, keepdim = True)
        getitem_14: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = var_mean_4[0]
        getitem_15: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_48, getitem_15);  getitem_15 = None
        add_19: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_4: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_13: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_14: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, arg28_1);  mul_13 = arg28_1 = None
        add_20: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_14, arg29_1);  mul_14 = arg29_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_49: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(add_20, [401408, 128]);  add_20 = None
        permute_19: "f32[128, 512][1, 128]cuda:0" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_6: "f32[401408, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg31_1, view_49, permute_19);  arg31_1 = view_49 = permute_19 = None
        view_50: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 3136, 512]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_15: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_50, 0.5)
        mul_16: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476);  view_50 = None
        erf_1: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_21: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_17: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, add_21);  mul_15 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_51: "f32[401408, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_17, [401408, 512]);  mul_17 = None
        permute_20: "f32[512, 128][1, 512]cuda:0" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_7: "f32[401408, 128][128, 1]cuda:0" = torch.ops.aten.addmm.default(arg33_1, view_51, permute_20);  arg33_1 = view_51 = permute_20 = None
        view_52: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [128, 3136, 128]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_22: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_48, view_52);  view_48 = view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_53: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_22, [128, 56, 56, 128]);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_54: "f32[128, 28, 2, 28, 2, 128][401408, 14336, 7168, 256, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_53, [128, 28, 2, 28, 2, 128]);  view_53 = None
        permute_21: "f32[128, 28, 28, 2, 2, 128][401408, 14336, 256, 128, 7168, 1]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 4, 2, 5]);  view_54 = None
        clone_22: "f32[128, 28, 28, 2, 2, 128][401408, 14336, 512, 256, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_55: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [128, 28, 28, 512]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_5 = torch.ops.aten.var_mean.correction(view_55, [3], correction = 0, keepdim = True)
        getitem_16: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_5[0]
        getitem_17: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_55, getitem_17);  view_55 = getitem_17 = None
        add_23: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_5: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_18: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_19: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, arg34_1);  mul_18 = arg34_1 = None
        add_24: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, arg35_1);  mul_19 = arg35_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_56: "f32[100352, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_24, [100352, 512]);  add_24 = None
        permute_22: "f32[512, 256][1, 512]cuda:0" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_56, permute_22);  view_56 = permute_22 = None
        view_57: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 28, 28, 256]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_6 = torch.ops.aten.var_mean.correction(view_57, [3], correction = 0, keepdim = True)
        getitem_18: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_6[0]
        getitem_19: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_8: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_57, getitem_19);  getitem_19 = None
        add_25: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_6: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_20: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_21: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, arg37_1);  mul_20 = arg37_1 = None
        add_26: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, arg38_1);  mul_21 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_58: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_26, [128, 4, 7, 4, 7, 256]);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_23: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 1792, 7168, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 1, 3, 2, 4, 5]);  view_58 = None
        clone_23: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_59: "f32[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [-1, 7, 7, 256]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_60: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [-1, 49, 256]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_61: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_60, [100352, 256]);  view_60 = None
        permute_24: "f32[256, 768][1, 256]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_8: "f32[100352, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg40_1, view_61, permute_24);  arg40_1 = view_61 = permute_24 = None
        view_62: "f32[2048, 49, 768][37632, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [2048, 49, 768]);  addmm_8 = None
        view_63: "f32[2048, 49, 3, 8, 32][37632, 768, 256, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [2048, 49, 3, 8, -1]);  view_62 = None
        permute_25: "f32[3, 2048, 8, 49, 32][256, 37632, 32, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_20: "f32[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_2[0]
        getitem_21: "f32[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_2[1]
        getitem_22: "f32[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_22: "f32[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_20, 0.1767766952966369);  getitem_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_8: "f32[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_22, [2048, 8, 49, 32]);  mul_22 = None
        clone_24: "f32[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_64: "f32[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [16384, 49, 32]);  clone_24 = None
        permute_26: "f32[2048, 8, 32, 49][37632, 32, 1, 768]cuda:0" = torch.ops.aten.permute.default(getitem_21, [0, 1, 3, 2]);  getitem_21 = None
        expand_9: "f32[2048, 8, 32, 49][37632, 32, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_26, [2048, 8, 32, 49]);  permute_26 = None
        clone_25: "f32[2048, 8, 32, 49][12544, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_65: "f32[16384, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [16384, 32, 49]);  clone_25 = None
        bmm_4: "f32[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_64, view_65);  view_64 = view_65 = None
        view_66: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [2048, 8, 49, 49]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_67: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg42_1, [-1]);  arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_6: "f32[2401, 8][8, 1]cuda:0" = torch.ops.aten.index.Tensor(arg41_1, [view_67]);  arg41_1 = view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_68: "f32[49, 49, 8][392, 8, 1]cuda:0" = torch.ops.aten.reshape.default(index_6, [49, 49, -1]);  index_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_27: "f32[8, 49, 49][1, 392, 8]cuda:0" = torch.ops.aten.permute.default(view_68, [2, 0, 1]);  view_68 = None
        clone_26: "f32[8, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_4: "f32[1, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_26, 0);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_27: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_66, unsqueeze_4);  view_66 = unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_2: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_27, [-1], True)
        sub_9: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_27, amax_2);  add_27 = amax_2 = None
        exp_2: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_10: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_2, [2048, 8, 49, 49]);  div_2 = None
        view_69: "f32[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_10, [16384, 49, 49]);  expand_10 = None
        expand_11: "f32[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = torch.ops.aten.expand.default(getitem_22, [2048, 8, 49, 32]);  getitem_22 = None
        clone_28: "f32[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_70: "f32[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [16384, 49, 32]);  clone_28 = None
        bmm_5: "f32[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_69, view_70);  view_69 = view_70 = None
        view_71: "f32[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [2048, 8, 49, 32]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_28: "f32[2048, 49, 8, 32][12544, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        clone_29: "f32[2048, 49, 8, 32][12544, 256, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_72: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [2048, 49, 256]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_73: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_72, [100352, 256]);  view_72 = None
        permute_29: "f32[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_9: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_73, permute_29);  arg44_1 = view_73 = permute_29 = None
        view_74: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [2048, 49, 256]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_75: "f32[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [-1, 7, 7, 256]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_76: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_75, [-1, 4, 4, 7, 7, 256]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_30: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 1792, 12544, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 1, 3, 2, 4, 5]);  view_76 = None
        clone_31: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_77: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [-1, 28, 28, 256]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_28: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_57, view_77);  view_57 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_78: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_28, [128, -1, 256]);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_7 = torch.ops.aten.var_mean.correction(view_78, [2], correction = 0, keepdim = True)
        getitem_23: "f32[128, 784, 1][784, 1, 1]cuda:0" = var_mean_7[0]
        getitem_24: "f32[128, 784, 1][784, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_10: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_78, getitem_24);  getitem_24 = None
        add_29: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_23, 1e-05);  getitem_23 = None
        rsqrt_7: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_23: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = rsqrt_7 = None
        mul_24: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, arg45_1);  mul_23 = arg45_1 = None
        add_30: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_24, arg46_1);  mul_24 = arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_79: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(add_30, [100352, 256]);  add_30 = None
        permute_31: "f32[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_10: "f32[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, view_79, permute_31);  arg48_1 = view_79 = permute_31 = None
        view_80: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 784, 1024]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_25: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_80, 0.5)
        mul_26: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476);  view_80 = None
        erf_2: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_31: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_27: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, add_31);  mul_25 = add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_81: "f32[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [100352, 1024]);  mul_27 = None
        permute_32: "f32[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_11: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(arg50_1, view_81, permute_32);  arg50_1 = view_81 = permute_32 = None
        view_82: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [128, 784, 256]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_32: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_78, view_82);  view_78 = view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_83: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_32, [128, 28, 28, 256]);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_8 = torch.ops.aten.var_mean.correction(view_83, [3], correction = 0, keepdim = True)
        getitem_25: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_8[0]
        getitem_26: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_11: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_83, getitem_26);  getitem_26 = None
        add_33: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_25, 1e-05);  getitem_25 = None
        rsqrt_8: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_28: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = rsqrt_8 = None
        mul_29: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, arg51_1);  mul_28 = arg51_1 = None
        add_34: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, arg52_1);  mul_29 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_4: "i64[28][1]cuda:0" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_35: "i64[28][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 3);  iota_4 = None
        fmod_4: "i64[28][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_35, 28);  add_35 = None
        index_7: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(add_34, [None, fmod_4]);  add_34 = fmod_4 = None
        iota_5: "i64[28][1]cuda:0" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_36: "i64[28][1]cuda:0" = torch.ops.aten.add.Tensor(iota_5, 3);  iota_5 = None
        fmod_5: "i64[28][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_36, 28);  add_36 = None
        index_8: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(index_7, [None, None, fmod_5]);  index_7 = fmod_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_84: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(index_8, [128, 4, 7, 4, 7, 256]);  index_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_33: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 1792, 7168, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_84, [0, 1, 3, 2, 4, 5]);  view_84 = None
        clone_34: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_85: "f32[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [-1, 7, 7, 256]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_86: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_85, [-1, 49, 256]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_87: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [100352, 256]);  view_86 = None
        permute_34: "f32[256, 768][1, 256]cuda:0" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_12: "f32[100352, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg55_1, view_87, permute_34);  arg55_1 = view_87 = permute_34 = None
        view_88: "f32[2048, 49, 768][37632, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [2048, 49, 768]);  addmm_12 = None
        view_89: "f32[2048, 49, 3, 8, 32][37632, 768, 256, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_88, [2048, 49, 3, 8, -1]);  view_88 = None
        permute_35: "f32[3, 2048, 8, 49, 32][256, 37632, 32, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_89, [2, 0, 3, 1, 4]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_35);  permute_35 = None
        getitem_27: "f32[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_3[0]
        getitem_28: "f32[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_3[1]
        getitem_29: "f32[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_30: "f32[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_27, 0.1767766952966369);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_12: "f32[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_30, [2048, 8, 49, 32]);  mul_30 = None
        clone_35: "f32[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_90: "f32[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [16384, 49, 32]);  clone_35 = None
        permute_36: "f32[2048, 8, 32, 49][37632, 32, 1, 768]cuda:0" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_13: "f32[2048, 8, 32, 49][37632, 32, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_36, [2048, 8, 32, 49]);  permute_36 = None
        clone_36: "f32[2048, 8, 32, 49][12544, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_91: "f32[16384, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [16384, 32, 49]);  clone_36 = None
        bmm_6: "f32[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_90, view_91);  view_90 = view_91 = None
        view_92: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [2048, 8, 49, 49]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_93: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg57_1, [-1]);  arg57_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_9: "f32[2401, 8][8, 1]cuda:0" = torch.ops.aten.index.Tensor(arg56_1, [view_93]);  arg56_1 = view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_94: "f32[49, 49, 8][392, 8, 1]cuda:0" = torch.ops.aten.reshape.default(index_9, [49, 49, -1]);  index_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_37: "f32[8, 49, 49][1, 392, 8]cuda:0" = torch.ops.aten.permute.default(view_94, [2, 0, 1]);  view_94 = None
        clone_37: "f32[8, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_5: "f32[1, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_37, 0);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_37: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_92, unsqueeze_5);  view_92 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_95: "f32[128, 16, 8, 49, 49][307328, 19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_37, [-1, 16, 8, 49, 49]);  add_37 = None
        unsqueeze_6: "f32[16, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg53_1, 1);  arg53_1 = None
        unsqueeze_7: "f32[1, 16, 1, 49, 49][38416, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 0);  unsqueeze_6 = None
        add_38: "f32[128, 16, 8, 49, 49][307328, 19208, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_95, unsqueeze_7);  view_95 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_96: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_38, [-1, 8, 49, 49]);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_3: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_96, [-1], True)
        sub_12: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_96, amax_3);  view_96 = amax_3 = None
        exp_3: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_4: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_14: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_3, [2048, 8, 49, 49]);  div_3 = None
        view_97: "f32[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_14, [16384, 49, 49]);  expand_14 = None
        expand_15: "f32[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = torch.ops.aten.expand.default(getitem_29, [2048, 8, 49, 32]);  getitem_29 = None
        clone_39: "f32[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_98: "f32[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [16384, 49, 32]);  clone_39 = None
        bmm_7: "f32[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
        view_99: "f32[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [2048, 8, 49, 32]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_38: "f32[2048, 49, 8, 32][12544, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        clone_40: "f32[2048, 49, 8, 32][12544, 256, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_100: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [2048, 49, 256]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_101: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_100, [100352, 256]);  view_100 = None
        permute_39: "f32[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_13: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(arg59_1, view_101, permute_39);  arg59_1 = view_101 = permute_39 = None
        view_102: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [2048, 49, 256]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_103: "f32[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [-1, 7, 7, 256]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_104: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [-1, 4, 4, 7, 7, 256]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_40: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 1792, 12544, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_104, [0, 1, 3, 2, 4, 5]);  view_104 = None
        clone_42: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_105: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [-1, 28, 28, 256]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_6: "i64[28][1]cuda:0" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_39: "i64[28][1]cuda:0" = torch.ops.aten.add.Tensor(iota_6, 25);  iota_6 = None
        fmod_6: "i64[28][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_39, 28);  add_39 = None
        index_10: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(view_105, [None, fmod_6]);  view_105 = fmod_6 = None
        iota_7: "i64[28][1]cuda:0" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_40: "i64[28][1]cuda:0" = torch.ops.aten.add.Tensor(iota_7, 25);  iota_7 = None
        fmod_7: "i64[28][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_40, 28);  add_40 = None
        index_11: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(index_10, [None, None, fmod_7]);  index_10 = fmod_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_41: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_83, index_11);  view_83 = index_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_106: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_41, [128, -1, 256]);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_9 = torch.ops.aten.var_mean.correction(view_106, [2], correction = 0, keepdim = True)
        getitem_30: "f32[128, 784, 1][784, 1, 1]cuda:0" = var_mean_9[0]
        getitem_31: "f32[128, 784, 1][784, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_13: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_106, getitem_31);  getitem_31 = None
        add_42: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_9: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_31: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = rsqrt_9 = None
        mul_32: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, arg60_1);  mul_31 = arg60_1 = None
        add_43: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, arg61_1);  mul_32 = arg61_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_107: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(add_43, [100352, 256]);  add_43 = None
        permute_41: "f32[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_14: "f32[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg63_1, view_107, permute_41);  arg63_1 = view_107 = permute_41 = None
        view_108: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 784, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_33: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, 0.5)
        mul_34: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476);  view_108 = None
        erf_3: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_44: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_35: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, add_44);  mul_33 = add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_109: "f32[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_35, [100352, 1024]);  mul_35 = None
        permute_42: "f32[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_15: "f32[100352, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(arg65_1, view_109, permute_42);  arg65_1 = view_109 = permute_42 = None
        view_110: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [128, 784, 256]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_45: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_106, view_110);  view_106 = view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_111: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_45, [128, 28, 28, 256]);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_112: "f32[128, 14, 2, 14, 2, 256][200704, 14336, 7168, 512, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [128, 14, 2, 14, 2, 256]);  view_111 = None
        permute_43: "f32[128, 14, 14, 2, 2, 256][200704, 14336, 512, 256, 7168, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 4, 2, 5]);  view_112 = None
        clone_45: "f32[128, 14, 14, 2, 2, 256][200704, 14336, 1024, 512, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_113: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [128, 14, 14, 1024]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_10 = torch.ops.aten.var_mean.correction(view_113, [3], correction = 0, keepdim = True)
        getitem_32: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_10[0]
        getitem_33: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_14: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_113, getitem_33);  view_113 = getitem_33 = None
        add_46: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_10: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_36: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = rsqrt_10 = None
        mul_37: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg66_1);  mul_36 = arg66_1 = None
        add_47: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg67_1);  mul_37 = arg67_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_114: "f32[25088, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_47, [25088, 1024]);  add_47 = None
        permute_44: "f32[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_1: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_114, permute_44);  view_114 = permute_44 = None
        view_115: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [128, 14, 14, 512]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_11 = torch.ops.aten.var_mean.correction(view_115, [3], correction = 0, keepdim = True)
        getitem_34: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_11[0]
        getitem_35: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_15: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_115, getitem_35);  getitem_35 = None
        add_48: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_11: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_38: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = rsqrt_11 = None
        mul_39: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, arg69_1);  mul_38 = arg69_1 = None
        add_49: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, arg70_1);  mul_39 = arg70_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_116: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_49, [128, 2, 7, 2, 7, 512]);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_45: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 1, 3, 2, 4, 5]);  view_116 = None
        clone_46: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_117: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [-1, 7, 7, 512]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_118: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [-1, 49, 512]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_119: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_118, [25088, 512]);  view_118 = None
        permute_46: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_16: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_119, permute_46);  arg72_1 = view_119 = permute_46 = None
        view_120: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [512, 49, 1536]);  addmm_16 = None
        view_121: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_120, [512, 49, 3, 16, -1]);  view_120 = None
        permute_47: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_121, [2, 0, 3, 1, 4]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
        getitem_36: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_4[0]
        getitem_37: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_4[1]
        getitem_38: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_40: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_36, 0.1767766952966369);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_16: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_40, [512, 16, 49, 32]);  mul_40 = None
        clone_47: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_122: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [8192, 49, 32]);  clone_47 = None
        permute_48: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_37, [0, 1, 3, 2]);  getitem_37 = None
        expand_17: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_48, [512, 16, 32, 49]);  permute_48 = None
        clone_48: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_123: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [8192, 32, 49]);  clone_48 = None
        bmm_8: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
        view_124: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [512, 16, 49, 49]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_125: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg74_1, [-1]);  arg74_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_12: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg73_1, [view_125]);  arg73_1 = view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_126: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_12, [49, 49, -1]);  index_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_49: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_126, [2, 0, 1]);  view_126 = None
        clone_49: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_8: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_49, 0);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_50: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_124, unsqueeze_8);  view_124 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_4: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_50, [-1], True)
        sub_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_50, amax_4);  add_50 = amax_4 = None
        exp_4: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_5: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_18: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_4, [512, 16, 49, 49]);  div_4 = None
        view_127: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_18, [8192, 49, 49]);  expand_18 = None
        expand_19: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_38, [512, 16, 49, 32]);  getitem_38 = None
        clone_51: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_128: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [8192, 49, 32]);  clone_51 = None
        bmm_9: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_127, view_128);  view_127 = view_128 = None
        view_129: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [512, 16, 49, 32]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_50: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_52: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_130: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [512, 49, 512]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_131: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_130, [25088, 512]);  view_130 = None
        permute_51: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_17: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg76_1, view_131, permute_51);  arg76_1 = view_131 = permute_51 = None
        view_132: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [512, 49, 512]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_133: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_132, [-1, 7, 7, 512]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_134: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [-1, 2, 2, 7, 7, 512]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_52: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 1, 3, 2, 4, 5]);  view_134 = None
        clone_54: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_135: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [-1, 14, 14, 512]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_51: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_115, view_135);  view_115 = view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_136: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_51, [128, -1, 512]);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_12 = torch.ops.aten.var_mean.correction(view_136, [2], correction = 0, keepdim = True)
        getitem_39: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_12[0]
        getitem_40: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_17: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_136, getitem_40);  getitem_40 = None
        add_52: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_12: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_41: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = rsqrt_12 = None
        mul_42: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, arg77_1);  mul_41 = arg77_1 = None
        add_53: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_42, arg78_1);  mul_42 = arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_137: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_53, [25088, 512]);  add_53 = None
        permute_53: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_18: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_137, permute_53);  arg80_1 = view_137 = permute_53 = None
        view_138: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 196, 2048]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_43: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_138, 0.5)
        mul_44: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
        erf_4: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_54: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_45: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, add_54);  mul_43 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_139: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_45, [25088, 2048]);  mul_45 = None
        permute_54: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_19: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg82_1, view_139, permute_54);  arg82_1 = view_139 = permute_54 = None
        view_140: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [128, 196, 512]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_55: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_136, view_140);  view_136 = view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_141: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_55, [128, 14, 14, 512]);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_13 = torch.ops.aten.var_mean.correction(view_141, [3], correction = 0, keepdim = True)
        getitem_41: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_13[0]
        getitem_42: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_18: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_141, getitem_42);  getitem_42 = None
        add_56: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
        rsqrt_13: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_46: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = rsqrt_13 = None
        mul_47: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, arg83_1);  mul_46 = arg83_1 = None
        add_57: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_47, arg84_1);  mul_47 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_8: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_58: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_8, 3);  iota_8 = None
        fmod_8: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_58, 14);  add_58 = None
        index_13: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_57, [None, fmod_8]);  add_57 = fmod_8 = None
        iota_9: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_59: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_9, 3);  iota_9 = None
        fmod_9: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_59, 14);  add_59 = None
        index_14: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_13, [None, None, fmod_9]);  index_13 = fmod_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_142: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_14, [128, 2, 7, 2, 7, 512]);  index_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_55: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 1, 3, 2, 4, 5]);  view_142 = None
        clone_57: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_143: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [-1, 7, 7, 512]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_144: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_143, [-1, 49, 512]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_145: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_144, [25088, 512]);  view_144 = None
        permute_56: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_20: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg87_1, view_145, permute_56);  arg87_1 = view_145 = permute_56 = None
        view_146: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [512, 49, 1536]);  addmm_20 = None
        view_147: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_146, [512, 49, 3, 16, -1]);  view_146 = None
        permute_57: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_147, [2, 0, 3, 1, 4]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_57);  permute_57 = None
        getitem_43: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_5[0]
        getitem_44: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_5[1]
        getitem_45: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_48: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_43, 0.1767766952966369);  getitem_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_20: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_48, [512, 16, 49, 32]);  mul_48 = None
        clone_58: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_148: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [8192, 49, 32]);  clone_58 = None
        permute_58: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_44, [0, 1, 3, 2]);  getitem_44 = None
        expand_21: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_58, [512, 16, 32, 49]);  permute_58 = None
        clone_59: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_149: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [8192, 32, 49]);  clone_59 = None
        bmm_10: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_148, view_149);  view_148 = view_149 = None
        view_150: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [512, 16, 49, 49]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_151: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg89_1, [-1]);  arg89_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_15: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg88_1, [view_151]);  arg88_1 = view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_152: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_15, [49, 49, -1]);  index_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_59: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_152, [2, 0, 1]);  view_152 = None
        clone_60: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_9: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_60, 0);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_60: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_150, unsqueeze_9);  view_150 = unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_153: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_60, [-1, 4, 16, 49, 49]);  add_60 = None
        unsqueeze_10: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg85_1, 1);  arg85_1 = None
        unsqueeze_11: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 0);  unsqueeze_10 = None
        add_61: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_153, unsqueeze_11);  view_153 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_154: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_61, [-1, 16, 49, 49]);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_5: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_154, [-1], True)
        sub_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_154, amax_5);  view_154 = amax_5 = None
        exp_5: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_6: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_22: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_5, [512, 16, 49, 49]);  div_5 = None
        view_155: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_22, [8192, 49, 49]);  expand_22 = None
        expand_23: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_45, [512, 16, 49, 32]);  getitem_45 = None
        clone_62: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_156: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [8192, 49, 32]);  clone_62 = None
        bmm_11: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_155, view_156);  view_155 = view_156 = None
        view_157: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [512, 16, 49, 32]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_60: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        clone_63: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_158: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [512, 49, 512]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_159: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_158, [25088, 512]);  view_158 = None
        permute_61: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_21: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg91_1, view_159, permute_61);  arg91_1 = view_159 = permute_61 = None
        view_160: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [512, 49, 512]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_161: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_160, [-1, 7, 7, 512]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_162: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [-1, 2, 2, 7, 7, 512]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_62: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 1, 3, 2, 4, 5]);  view_162 = None
        clone_65: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_163: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [-1, 14, 14, 512]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_10: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_62: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_10, 11);  iota_10 = None
        fmod_10: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_62, 14);  add_62 = None
        index_16: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_163, [None, fmod_10]);  view_163 = fmod_10 = None
        iota_11: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_63: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_11, 11);  iota_11 = None
        fmod_11: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_63, 14);  add_63 = None
        index_17: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_16, [None, None, fmod_11]);  index_16 = fmod_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_64: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_141, index_17);  view_141 = index_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_164: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_64, [128, -1, 512]);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_14 = torch.ops.aten.var_mean.correction(view_164, [2], correction = 0, keepdim = True)
        getitem_46: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_14[0]
        getitem_47: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_20: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_164, getitem_47);  getitem_47 = None
        add_65: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_14: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_49: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = rsqrt_14 = None
        mul_50: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, arg92_1);  mul_49 = arg92_1 = None
        add_66: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, arg93_1);  mul_50 = arg93_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_165: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_66, [25088, 512]);  add_66 = None
        permute_63: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_22: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg95_1, view_165, permute_63);  arg95_1 = view_165 = permute_63 = None
        view_166: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 196, 2048]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_51: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_166, 0.5)
        mul_52: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_166, 0.7071067811865476);  view_166 = None
        erf_5: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_67: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_53: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, add_67);  mul_51 = add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_167: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_53, [25088, 2048]);  mul_53 = None
        permute_64: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_23: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg97_1, view_167, permute_64);  arg97_1 = view_167 = permute_64 = None
        view_168: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [128, 196, 512]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_68: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_164, view_168);  view_164 = view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_169: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_68, [128, 14, 14, 512]);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_15 = torch.ops.aten.var_mean.correction(view_169, [3], correction = 0, keepdim = True)
        getitem_48: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_15[0]
        getitem_49: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_21: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_169, getitem_49);  getitem_49 = None
        add_69: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_15: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_54: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = rsqrt_15 = None
        mul_55: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, arg98_1);  mul_54 = arg98_1 = None
        add_70: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, arg99_1);  mul_55 = arg99_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_170: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_70, [128, 2, 7, 2, 7, 512]);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_65: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_170, [0, 1, 3, 2, 4, 5]);  view_170 = None
        clone_68: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_171: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [-1, 7, 7, 512]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_172: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_171, [-1, 49, 512]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_173: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_172, [25088, 512]);  view_172 = None
        permute_66: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_24: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg101_1, view_173, permute_66);  arg101_1 = view_173 = permute_66 = None
        view_174: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [512, 49, 1536]);  addmm_24 = None
        view_175: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_174, [512, 49, 3, 16, -1]);  view_174 = None
        permute_67: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_175, [2, 0, 3, 1, 4]);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
        getitem_50: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_6[0]
        getitem_51: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_6[1]
        getitem_52: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_56: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_50, 0.1767766952966369);  getitem_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_24: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_56, [512, 16, 49, 32]);  mul_56 = None
        clone_69: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_176: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [8192, 49, 32]);  clone_69 = None
        permute_68: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_51, [0, 1, 3, 2]);  getitem_51 = None
        expand_25: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_68, [512, 16, 32, 49]);  permute_68 = None
        clone_70: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_177: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [8192, 32, 49]);  clone_70 = None
        bmm_12: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_176, view_177);  view_176 = view_177 = None
        view_178: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [512, 16, 49, 49]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_179: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg103_1, [-1]);  arg103_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_18: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg102_1, [view_179]);  arg102_1 = view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_180: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_18, [49, 49, -1]);  index_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_69: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_180, [2, 0, 1]);  view_180 = None
        clone_71: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_12: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_71, 0);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_71: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_178, unsqueeze_12);  view_178 = unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_6: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_71, [-1], True)
        sub_22: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_71, amax_6);  add_71 = amax_6 = None
        exp_6: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_7: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_26: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_6, [512, 16, 49, 49]);  div_6 = None
        view_181: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_26, [8192, 49, 49]);  expand_26 = None
        expand_27: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_52, [512, 16, 49, 32]);  getitem_52 = None
        clone_73: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_182: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [8192, 49, 32]);  clone_73 = None
        bmm_13: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_181, view_182);  view_181 = view_182 = None
        view_183: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [512, 16, 49, 32]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_70: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        clone_74: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_184: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [512, 49, 512]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_185: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_184, [25088, 512]);  view_184 = None
        permute_71: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_25: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg105_1, view_185, permute_71);  arg105_1 = view_185 = permute_71 = None
        view_186: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [512, 49, 512]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_187: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_186, [-1, 7, 7, 512]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_188: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_187, [-1, 2, 2, 7, 7, 512]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_72: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 1, 3, 2, 4, 5]);  view_188 = None
        clone_76: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_189: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [-1, 14, 14, 512]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_72: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_169, view_189);  view_169 = view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_190: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_72, [128, -1, 512]);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_16 = torch.ops.aten.var_mean.correction(view_190, [2], correction = 0, keepdim = True)
        getitem_53: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_16[0]
        getitem_54: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_23: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_190, getitem_54);  getitem_54 = None
        add_73: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_16: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_57: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = rsqrt_16 = None
        mul_58: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg106_1);  mul_57 = arg106_1 = None
        add_74: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg107_1);  mul_58 = arg107_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_191: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_74, [25088, 512]);  add_74 = None
        permute_73: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_26: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg109_1, view_191, permute_73);  arg109_1 = view_191 = permute_73 = None
        view_192: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 196, 2048]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_59: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_192, 0.5)
        mul_60: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_192, 0.7071067811865476);  view_192 = None
        erf_6: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_75: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_61: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, add_75);  mul_59 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_193: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_61, [25088, 2048]);  mul_61 = None
        permute_74: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_27: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg111_1, view_193, permute_74);  arg111_1 = view_193 = permute_74 = None
        view_194: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [128, 196, 512]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_76: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_190, view_194);  view_190 = view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_195: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_76, [128, 14, 14, 512]);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_17 = torch.ops.aten.var_mean.correction(view_195, [3], correction = 0, keepdim = True)
        getitem_55: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_17[0]
        getitem_56: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_24: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_195, getitem_56);  getitem_56 = None
        add_77: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_17: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_62: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = rsqrt_17 = None
        mul_63: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, arg112_1);  mul_62 = arg112_1 = None
        add_78: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, arg113_1);  mul_63 = arg113_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_12: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_79: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_12, 3);  iota_12 = None
        fmod_12: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_79, 14);  add_79 = None
        index_19: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_78, [None, fmod_12]);  add_78 = fmod_12 = None
        iota_13: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_80: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_13, 3);  iota_13 = None
        fmod_13: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_80, 14);  add_80 = None
        index_20: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_19, [None, None, fmod_13]);  index_19 = fmod_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_196: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_20, [128, 2, 7, 2, 7, 512]);  index_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_75: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_196, [0, 1, 3, 2, 4, 5]);  view_196 = None
        clone_79: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_197: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [-1, 7, 7, 512]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_198: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_197, [-1, 49, 512]);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_199: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_198, [25088, 512]);  view_198 = None
        permute_76: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_28: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg116_1, view_199, permute_76);  arg116_1 = view_199 = permute_76 = None
        view_200: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [512, 49, 1536]);  addmm_28 = None
        view_201: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_200, [512, 49, 3, 16, -1]);  view_200 = None
        permute_77: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_77);  permute_77 = None
        getitem_57: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_7[0]
        getitem_58: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_7[1]
        getitem_59: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_64: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_57, 0.1767766952966369);  getitem_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_28: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_64, [512, 16, 49, 32]);  mul_64 = None
        clone_80: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_202: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [8192, 49, 32]);  clone_80 = None
        permute_78: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_58, [0, 1, 3, 2]);  getitem_58 = None
        expand_29: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_78, [512, 16, 32, 49]);  permute_78 = None
        clone_81: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_203: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [8192, 32, 49]);  clone_81 = None
        bmm_14: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_202, view_203);  view_202 = view_203 = None
        view_204: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [512, 16, 49, 49]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_205: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg118_1, [-1]);  arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_21: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg117_1, [view_205]);  arg117_1 = view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_206: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_21, [49, 49, -1]);  index_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_79: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_206, [2, 0, 1]);  view_206 = None
        clone_82: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_13: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_82, 0);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_81: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_204, unsqueeze_13);  view_204 = unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_207: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_81, [-1, 4, 16, 49, 49]);  add_81 = None
        unsqueeze_14: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg114_1, 1);  arg114_1 = None
        unsqueeze_15: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 0);  unsqueeze_14 = None
        add_82: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_207, unsqueeze_15);  view_207 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_208: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_82, [-1, 16, 49, 49]);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_7: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_208, [-1], True)
        sub_25: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_208, amax_7);  view_208 = amax_7 = None
        exp_7: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_8: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_30: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_7, [512, 16, 49, 49]);  div_7 = None
        view_209: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_30, [8192, 49, 49]);  expand_30 = None
        expand_31: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_59, [512, 16, 49, 32]);  getitem_59 = None
        clone_84: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_210: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [8192, 49, 32]);  clone_84 = None
        bmm_15: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_209, view_210);  view_209 = view_210 = None
        view_211: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [512, 16, 49, 32]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_80: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
        clone_85: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_212: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [512, 49, 512]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_213: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_212, [25088, 512]);  view_212 = None
        permute_81: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_29: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg120_1, view_213, permute_81);  arg120_1 = view_213 = permute_81 = None
        view_214: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [512, 49, 512]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_215: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_214, [-1, 7, 7, 512]);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_216: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_215, [-1, 2, 2, 7, 7, 512]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_82: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_216, [0, 1, 3, 2, 4, 5]);  view_216 = None
        clone_87: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_217: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [-1, 14, 14, 512]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_14: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_83: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_14, 11);  iota_14 = None
        fmod_14: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_83, 14);  add_83 = None
        index_22: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_217, [None, fmod_14]);  view_217 = fmod_14 = None
        iota_15: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_84: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_15, 11);  iota_15 = None
        fmod_15: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_84, 14);  add_84 = None
        index_23: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_22, [None, None, fmod_15]);  index_22 = fmod_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_85: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_195, index_23);  view_195 = index_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_218: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_85, [128, -1, 512]);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_18 = torch.ops.aten.var_mean.correction(view_218, [2], correction = 0, keepdim = True)
        getitem_60: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_18[0]
        getitem_61: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_26: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_218, getitem_61);  getitem_61 = None
        add_86: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_18: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_65: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = rsqrt_18 = None
        mul_66: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, arg121_1);  mul_65 = arg121_1 = None
        add_87: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_66, arg122_1);  mul_66 = arg122_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_219: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_87, [25088, 512]);  add_87 = None
        permute_83: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_30: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg124_1, view_219, permute_83);  arg124_1 = view_219 = permute_83 = None
        view_220: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 196, 2048]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_67: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_220, 0.5)
        mul_68: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_220, 0.7071067811865476);  view_220 = None
        erf_7: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_88: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_69: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, add_88);  mul_67 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_221: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_69, [25088, 2048]);  mul_69 = None
        permute_84: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_31: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg126_1, view_221, permute_84);  arg126_1 = view_221 = permute_84 = None
        view_222: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [128, 196, 512]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_89: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_218, view_222);  view_218 = view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_223: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_89, [128, 14, 14, 512]);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_19 = torch.ops.aten.var_mean.correction(view_223, [3], correction = 0, keepdim = True)
        getitem_62: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_19[0]
        getitem_63: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_27: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_223, getitem_63);  getitem_63 = None
        add_90: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_19: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_70: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = rsqrt_19 = None
        mul_71: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, arg127_1);  mul_70 = arg127_1 = None
        add_91: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, arg128_1);  mul_71 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_224: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_91, [128, 2, 7, 2, 7, 512]);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_85: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_224, [0, 1, 3, 2, 4, 5]);  view_224 = None
        clone_90: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        view_225: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [-1, 7, 7, 512]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_226: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [-1, 49, 512]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_227: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_226, [25088, 512]);  view_226 = None
        permute_86: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_32: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg130_1, view_227, permute_86);  arg130_1 = view_227 = permute_86 = None
        view_228: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [512, 49, 1536]);  addmm_32 = None
        view_229: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_228, [512, 49, 3, 16, -1]);  view_228 = None
        permute_87: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_229, [2, 0, 3, 1, 4]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_87);  permute_87 = None
        getitem_64: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_8[0]
        getitem_65: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_8[1]
        getitem_66: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_72: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_64, 0.1767766952966369);  getitem_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_32: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_72, [512, 16, 49, 32]);  mul_72 = None
        clone_91: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_230: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [8192, 49, 32]);  clone_91 = None
        permute_88: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_65, [0, 1, 3, 2]);  getitem_65 = None
        expand_33: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_88, [512, 16, 32, 49]);  permute_88 = None
        clone_92: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_231: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [8192, 32, 49]);  clone_92 = None
        bmm_16: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_230, view_231);  view_230 = view_231 = None
        view_232: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [512, 16, 49, 49]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_233: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg132_1, [-1]);  arg132_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_24: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg131_1, [view_233]);  arg131_1 = view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_234: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_24, [49, 49, -1]);  index_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_89: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_234, [2, 0, 1]);  view_234 = None
        clone_93: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_16: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_93, 0);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_92: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_232, unsqueeze_16);  view_232 = unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_8: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_92, [-1], True)
        sub_28: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_92, amax_8);  add_92 = amax_8 = None
        exp_8: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_9: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_34: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_8, [512, 16, 49, 49]);  div_8 = None
        view_235: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_34, [8192, 49, 49]);  expand_34 = None
        expand_35: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_66, [512, 16, 49, 32]);  getitem_66 = None
        clone_95: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_236: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [8192, 49, 32]);  clone_95 = None
        bmm_17: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_235, view_236);  view_235 = view_236 = None
        view_237: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [512, 16, 49, 32]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_90: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        clone_96: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_238: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [512, 49, 512]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_239: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_238, [25088, 512]);  view_238 = None
        permute_91: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_33: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg134_1, view_239, permute_91);  arg134_1 = view_239 = permute_91 = None
        view_240: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [512, 49, 512]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_241: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_240, [-1, 7, 7, 512]);  view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_242: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_241, [-1, 2, 2, 7, 7, 512]);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_92: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_242, [0, 1, 3, 2, 4, 5]);  view_242 = None
        clone_98: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_243: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [-1, 14, 14, 512]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_93: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_223, view_243);  view_223 = view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_244: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_93, [128, -1, 512]);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_20 = torch.ops.aten.var_mean.correction(view_244, [2], correction = 0, keepdim = True)
        getitem_67: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_20[0]
        getitem_68: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_29: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_244, getitem_68);  getitem_68 = None
        add_94: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
        rsqrt_20: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_73: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = rsqrt_20 = None
        mul_74: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg135_1);  mul_73 = arg135_1 = None
        add_95: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg136_1);  mul_74 = arg136_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_245: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_95, [25088, 512]);  add_95 = None
        permute_93: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_34: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg138_1, view_245, permute_93);  arg138_1 = view_245 = permute_93 = None
        view_246: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 196, 2048]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_75: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_246, 0.5)
        mul_76: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_246, 0.7071067811865476);  view_246 = None
        erf_8: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_96: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_77: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, add_96);  mul_75 = add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_247: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_77, [25088, 2048]);  mul_77 = None
        permute_94: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_35: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_247, permute_94);  arg140_1 = view_247 = permute_94 = None
        view_248: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [128, 196, 512]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_97: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_244, view_248);  view_244 = view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_249: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_97, [128, 14, 14, 512]);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_21 = torch.ops.aten.var_mean.correction(view_249, [3], correction = 0, keepdim = True)
        getitem_69: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_21[0]
        getitem_70: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_30: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_249, getitem_70);  getitem_70 = None
        add_98: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_69, 1e-05);  getitem_69 = None
        rsqrt_21: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_78: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_21);  sub_30 = rsqrt_21 = None
        mul_79: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, arg141_1);  mul_78 = arg141_1 = None
        add_99: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, arg142_1);  mul_79 = arg142_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_16: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_100: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_16, 3);  iota_16 = None
        fmod_16: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_100, 14);  add_100 = None
        index_25: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_99, [None, fmod_16]);  add_99 = fmod_16 = None
        iota_17: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_101: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_17, 3);  iota_17 = None
        fmod_17: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_101, 14);  add_101 = None
        index_26: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_25, [None, None, fmod_17]);  index_25 = fmod_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_250: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_26, [128, 2, 7, 2, 7, 512]);  index_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_95: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 1, 3, 2, 4, 5]);  view_250 = None
        clone_101: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_251: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [-1, 7, 7, 512]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_252: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [-1, 49, 512]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_253: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_252, [25088, 512]);  view_252 = None
        permute_96: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_36: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg145_1, view_253, permute_96);  arg145_1 = view_253 = permute_96 = None
        view_254: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [512, 49, 1536]);  addmm_36 = None
        view_255: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_254, [512, 49, 3, 16, -1]);  view_254 = None
        permute_97: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_255, [2, 0, 3, 1, 4]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_9 = torch.ops.aten.unbind.int(permute_97);  permute_97 = None
        getitem_71: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_9[0]
        getitem_72: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_9[1]
        getitem_73: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_9[2];  unbind_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_80: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_71, 0.1767766952966369);  getitem_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_36: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_80, [512, 16, 49, 32]);  mul_80 = None
        clone_102: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_256: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [8192, 49, 32]);  clone_102 = None
        permute_98: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_72, [0, 1, 3, 2]);  getitem_72 = None
        expand_37: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_98, [512, 16, 32, 49]);  permute_98 = None
        clone_103: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_257: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [8192, 32, 49]);  clone_103 = None
        bmm_18: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_256, view_257);  view_256 = view_257 = None
        view_258: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [512, 16, 49, 49]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_259: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg147_1, [-1]);  arg147_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_27: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg146_1, [view_259]);  arg146_1 = view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_260: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_27, [49, 49, -1]);  index_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_99: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_260, [2, 0, 1]);  view_260 = None
        clone_104: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_17: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_104, 0);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_102: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_258, unsqueeze_17);  view_258 = unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_261: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_102, [-1, 4, 16, 49, 49]);  add_102 = None
        unsqueeze_18: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg143_1, 1);  arg143_1 = None
        unsqueeze_19: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 0);  unsqueeze_18 = None
        add_103: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_261, unsqueeze_19);  view_261 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_262: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_103, [-1, 16, 49, 49]);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_9: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_262, [-1], True)
        sub_31: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_262, amax_9);  view_262 = amax_9 = None
        exp_9: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_10: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_38: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_9, [512, 16, 49, 49]);  div_9 = None
        view_263: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_38, [8192, 49, 49]);  expand_38 = None
        expand_39: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_73, [512, 16, 49, 32]);  getitem_73 = None
        clone_106: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_264: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [8192, 49, 32]);  clone_106 = None
        bmm_19: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_263, view_264);  view_263 = view_264 = None
        view_265: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [512, 16, 49, 32]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_100: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        clone_107: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_266: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [512, 49, 512]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_267: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_266, [25088, 512]);  view_266 = None
        permute_101: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_37: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg149_1, view_267, permute_101);  arg149_1 = view_267 = permute_101 = None
        view_268: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [512, 49, 512]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_269: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_268, [-1, 7, 7, 512]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_270: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_269, [-1, 2, 2, 7, 7, 512]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_102: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_270, [0, 1, 3, 2, 4, 5]);  view_270 = None
        clone_109: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_271: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [-1, 14, 14, 512]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_18: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_104: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_18, 11);  iota_18 = None
        fmod_18: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_104, 14);  add_104 = None
        index_28: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_271, [None, fmod_18]);  view_271 = fmod_18 = None
        iota_19: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_105: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_19, 11);  iota_19 = None
        fmod_19: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_105, 14);  add_105 = None
        index_29: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_28, [None, None, fmod_19]);  index_28 = fmod_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_106: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_249, index_29);  view_249 = index_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_272: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_106, [128, -1, 512]);  add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_22 = torch.ops.aten.var_mean.correction(view_272, [2], correction = 0, keepdim = True)
        getitem_74: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_22[0]
        getitem_75: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_32: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_272, getitem_75);  getitem_75 = None
        add_107: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_22: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_81: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = rsqrt_22 = None
        mul_82: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, arg150_1);  mul_81 = arg150_1 = None
        add_108: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, arg151_1);  mul_82 = arg151_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_273: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_108, [25088, 512]);  add_108 = None
        permute_103: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_38: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg153_1, view_273, permute_103);  arg153_1 = view_273 = permute_103 = None
        view_274: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 196, 2048]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_83: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_274, 0.5)
        mul_84: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476);  view_274 = None
        erf_9: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_109: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_85: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, add_109);  mul_83 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_275: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_85, [25088, 2048]);  mul_85 = None
        permute_104: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_39: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg155_1, view_275, permute_104);  arg155_1 = view_275 = permute_104 = None
        view_276: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [128, 196, 512]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_110: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_272, view_276);  view_272 = view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_277: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_110, [128, 14, 14, 512]);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_23 = torch.ops.aten.var_mean.correction(view_277, [3], correction = 0, keepdim = True)
        getitem_76: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_23[0]
        getitem_77: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_33: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_277, getitem_77);  getitem_77 = None
        add_111: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_23: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_86: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_23);  sub_33 = rsqrt_23 = None
        mul_87: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, arg156_1);  mul_86 = arg156_1 = None
        add_112: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, arg157_1);  mul_87 = arg157_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_278: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_112, [128, 2, 7, 2, 7, 512]);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_105: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_278, [0, 1, 3, 2, 4, 5]);  view_278 = None
        clone_112: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_279: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_112, [-1, 7, 7, 512]);  clone_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_280: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_279, [-1, 49, 512]);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_281: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_280, [25088, 512]);  view_280 = None
        permute_106: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_40: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg159_1, view_281, permute_106);  arg159_1 = view_281 = permute_106 = None
        view_282: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [512, 49, 1536]);  addmm_40 = None
        view_283: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_282, [512, 49, 3, 16, -1]);  view_282 = None
        permute_107: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_283, [2, 0, 3, 1, 4]);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_10 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
        getitem_78: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_10[0]
        getitem_79: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_10[1]
        getitem_80: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_10[2];  unbind_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_88: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_78, 0.1767766952966369);  getitem_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_40: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_88, [512, 16, 49, 32]);  mul_88 = None
        clone_113: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_284: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [8192, 49, 32]);  clone_113 = None
        permute_108: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_79, [0, 1, 3, 2]);  getitem_79 = None
        expand_41: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_108, [512, 16, 32, 49]);  permute_108 = None
        clone_114: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_285: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [8192, 32, 49]);  clone_114 = None
        bmm_20: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_284, view_285);  view_284 = view_285 = None
        view_286: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [512, 16, 49, 49]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_287: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg161_1, [-1]);  arg161_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_30: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg160_1, [view_287]);  arg160_1 = view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_288: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_30, [49, 49, -1]);  index_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_109: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_288, [2, 0, 1]);  view_288 = None
        clone_115: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_20: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_115, 0);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_113: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_286, unsqueeze_20);  view_286 = unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_10: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_113, [-1], True)
        sub_34: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_113, amax_10);  add_113 = amax_10 = None
        exp_10: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_11: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_42: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_10, [512, 16, 49, 49]);  div_10 = None
        view_289: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_42, [8192, 49, 49]);  expand_42 = None
        expand_43: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_80, [512, 16, 49, 32]);  getitem_80 = None
        clone_117: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_290: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [8192, 49, 32]);  clone_117 = None
        bmm_21: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_289, view_290);  view_289 = view_290 = None
        view_291: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [512, 16, 49, 32]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_110: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_118: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_292: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [512, 49, 512]);  clone_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_293: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_292, [25088, 512]);  view_292 = None
        permute_111: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_41: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg163_1, view_293, permute_111);  arg163_1 = view_293 = permute_111 = None
        view_294: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [512, 49, 512]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_295: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_294, [-1, 7, 7, 512]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_296: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [-1, 2, 2, 7, 7, 512]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_112: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 1, 3, 2, 4, 5]);  view_296 = None
        clone_120: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_297: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [-1, 14, 14, 512]);  clone_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_114: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_277, view_297);  view_277 = view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_298: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_114, [128, -1, 512]);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_24 = torch.ops.aten.var_mean.correction(view_298, [2], correction = 0, keepdim = True)
        getitem_81: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_24[0]
        getitem_82: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        sub_35: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_298, getitem_82);  getitem_82 = None
        add_115: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_81, 1e-05);  getitem_81 = None
        rsqrt_24: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_89: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = rsqrt_24 = None
        mul_90: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, arg164_1);  mul_89 = arg164_1 = None
        add_116: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, arg165_1);  mul_90 = arg165_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_299: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_116, [25088, 512]);  add_116 = None
        permute_113: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_42: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg167_1, view_299, permute_113);  arg167_1 = view_299 = permute_113 = None
        view_300: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 196, 2048]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_91: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_300, 0.5)
        mul_92: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_300, 0.7071067811865476);  view_300 = None
        erf_10: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_117: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_93: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, add_117);  mul_91 = add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_301: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_93, [25088, 2048]);  mul_93 = None
        permute_114: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_43: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg169_1, view_301, permute_114);  arg169_1 = view_301 = permute_114 = None
        view_302: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [128, 196, 512]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_118: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_298, view_302);  view_298 = view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_303: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_118, [128, 14, 14, 512]);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_25 = torch.ops.aten.var_mean.correction(view_303, [3], correction = 0, keepdim = True)
        getitem_83: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_25[0]
        getitem_84: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        sub_36: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_303, getitem_84);  getitem_84 = None
        add_119: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_83, 1e-05);  getitem_83 = None
        rsqrt_25: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_94: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_25);  sub_36 = rsqrt_25 = None
        mul_95: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, arg170_1);  mul_94 = arg170_1 = None
        add_120: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, arg171_1);  mul_95 = arg171_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_20: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_121: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_20, 3);  iota_20 = None
        fmod_20: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_121, 14);  add_121 = None
        index_31: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_120, [None, fmod_20]);  add_120 = fmod_20 = None
        iota_21: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_122: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_21, 3);  iota_21 = None
        fmod_21: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_122, 14);  add_122 = None
        index_32: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_31, [None, None, fmod_21]);  index_31 = fmod_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_304: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_32, [128, 2, 7, 2, 7, 512]);  index_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_115: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_304, [0, 1, 3, 2, 4, 5]);  view_304 = None
        clone_123: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_305: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [-1, 7, 7, 512]);  clone_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_306: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_305, [-1, 49, 512]);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_307: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_306, [25088, 512]);  view_306 = None
        permute_116: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_44: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg174_1, view_307, permute_116);  arg174_1 = view_307 = permute_116 = None
        view_308: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [512, 49, 1536]);  addmm_44 = None
        view_309: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_308, [512, 49, 3, 16, -1]);  view_308 = None
        permute_117: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_309, [2, 0, 3, 1, 4]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_11 = torch.ops.aten.unbind.int(permute_117);  permute_117 = None
        getitem_85: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_11[0]
        getitem_86: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_11[1]
        getitem_87: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_11[2];  unbind_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_96: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_85, 0.1767766952966369);  getitem_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_44: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_96, [512, 16, 49, 32]);  mul_96 = None
        clone_124: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_310: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [8192, 49, 32]);  clone_124 = None
        permute_118: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_86, [0, 1, 3, 2]);  getitem_86 = None
        expand_45: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_118, [512, 16, 32, 49]);  permute_118 = None
        clone_125: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_311: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_125, [8192, 32, 49]);  clone_125 = None
        bmm_22: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_310, view_311);  view_310 = view_311 = None
        view_312: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [512, 16, 49, 49]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_313: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg176_1, [-1]);  arg176_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_33: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg175_1, [view_313]);  arg175_1 = view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_314: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_33, [49, 49, -1]);  index_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_119: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_314, [2, 0, 1]);  view_314 = None
        clone_126: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_21: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_126, 0);  clone_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_123: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_312, unsqueeze_21);  view_312 = unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_315: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_123, [-1, 4, 16, 49, 49]);  add_123 = None
        unsqueeze_22: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg172_1, 1);  arg172_1 = None
        unsqueeze_23: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 0);  unsqueeze_22 = None
        add_124: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_315, unsqueeze_23);  view_315 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_316: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_124, [-1, 16, 49, 49]);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_11: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_316, [-1], True)
        sub_37: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_316, amax_11);  view_316 = amax_11 = None
        exp_11: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_12: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_46: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_11, [512, 16, 49, 49]);  div_11 = None
        view_317: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_46, [8192, 49, 49]);  expand_46 = None
        expand_47: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_87, [512, 16, 49, 32]);  getitem_87 = None
        clone_128: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_318: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [8192, 49, 32]);  clone_128 = None
        bmm_23: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_317, view_318);  view_317 = view_318 = None
        view_319: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [512, 16, 49, 32]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_120: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_129: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        view_320: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [512, 49, 512]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_321: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_320, [25088, 512]);  view_320 = None
        permute_121: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_45: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg178_1, view_321, permute_121);  arg178_1 = view_321 = permute_121 = None
        view_322: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [512, 49, 512]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_323: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_322, [-1, 7, 7, 512]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_324: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_323, [-1, 2, 2, 7, 7, 512]);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_122: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_324, [0, 1, 3, 2, 4, 5]);  view_324 = None
        clone_131: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_325: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [-1, 14, 14, 512]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_22: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_125: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_22, 11);  iota_22 = None
        fmod_22: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_125, 14);  add_125 = None
        index_34: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_325, [None, fmod_22]);  view_325 = fmod_22 = None
        iota_23: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_126: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_23, 11);  iota_23 = None
        fmod_23: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_126, 14);  add_126 = None
        index_35: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_34, [None, None, fmod_23]);  index_34 = fmod_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_127: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_303, index_35);  view_303 = index_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_326: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_127, [128, -1, 512]);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_26 = torch.ops.aten.var_mean.correction(view_326, [2], correction = 0, keepdim = True)
        getitem_88: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_26[0]
        getitem_89: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        sub_38: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_326, getitem_89);  getitem_89 = None
        add_128: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_26: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_97: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = rsqrt_26 = None
        mul_98: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, arg179_1);  mul_97 = arg179_1 = None
        add_129: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_98, arg180_1);  mul_98 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_327: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_129, [25088, 512]);  add_129 = None
        permute_123: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_46: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg182_1, view_327, permute_123);  arg182_1 = view_327 = permute_123 = None
        view_328: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 196, 2048]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_99: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_328, 0.5)
        mul_100: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_328, 0.7071067811865476);  view_328 = None
        erf_11: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_130: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_101: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, add_130);  mul_99 = add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_329: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_101, [25088, 2048]);  mul_101 = None
        permute_124: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_47: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg184_1, view_329, permute_124);  arg184_1 = view_329 = permute_124 = None
        view_330: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [128, 196, 512]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_131: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_326, view_330);  view_326 = view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_331: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_131, [128, 14, 14, 512]);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_27 = torch.ops.aten.var_mean.correction(view_331, [3], correction = 0, keepdim = True)
        getitem_90: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_27[0]
        getitem_91: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        sub_39: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_331, getitem_91);  getitem_91 = None
        add_132: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_27: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_102: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_27);  sub_39 = rsqrt_27 = None
        mul_103: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, arg185_1);  mul_102 = arg185_1 = None
        add_133: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_103, arg186_1);  mul_103 = arg186_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_332: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_133, [128, 2, 7, 2, 7, 512]);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_125: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 1, 3, 2, 4, 5]);  view_332 = None
        clone_134: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_333: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_134, [-1, 7, 7, 512]);  clone_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_334: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_333, [-1, 49, 512]);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_335: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_334, [25088, 512]);  view_334 = None
        permute_126: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_48: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg188_1, view_335, permute_126);  arg188_1 = view_335 = permute_126 = None
        view_336: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [512, 49, 1536]);  addmm_48 = None
        view_337: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_336, [512, 49, 3, 16, -1]);  view_336 = None
        permute_127: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_337, [2, 0, 3, 1, 4]);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_12 = torch.ops.aten.unbind.int(permute_127);  permute_127 = None
        getitem_92: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_12[0]
        getitem_93: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_12[1]
        getitem_94: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_12[2];  unbind_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_104: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_92, 0.1767766952966369);  getitem_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_48: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_104, [512, 16, 49, 32]);  mul_104 = None
        clone_135: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_338: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [8192, 49, 32]);  clone_135 = None
        permute_128: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 1, 3, 2]);  getitem_93 = None
        expand_49: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_128, [512, 16, 32, 49]);  permute_128 = None
        clone_136: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_339: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [8192, 32, 49]);  clone_136 = None
        bmm_24: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_338, view_339);  view_338 = view_339 = None
        view_340: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [512, 16, 49, 49]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_341: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg190_1, [-1]);  arg190_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_36: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg189_1, [view_341]);  arg189_1 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_342: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_36, [49, 49, -1]);  index_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_129: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_342, [2, 0, 1]);  view_342 = None
        clone_137: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_24: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_137, 0);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_134: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_340, unsqueeze_24);  view_340 = unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_12: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_134, [-1], True)
        sub_40: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_134, amax_12);  add_134 = amax_12 = None
        exp_12: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_13: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_50: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_12, [512, 16, 49, 49]);  div_12 = None
        view_343: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_50, [8192, 49, 49]);  expand_50 = None
        expand_51: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_94, [512, 16, 49, 32]);  getitem_94 = None
        clone_139: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_344: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_139, [8192, 49, 32]);  clone_139 = None
        bmm_25: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_343, view_344);  view_343 = view_344 = None
        view_345: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [512, 16, 49, 32]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_130: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        clone_140: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_346: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_140, [512, 49, 512]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_347: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_346, [25088, 512]);  view_346 = None
        permute_131: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_49: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, view_347, permute_131);  arg192_1 = view_347 = permute_131 = None
        view_348: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [512, 49, 512]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_349: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_348, [-1, 7, 7, 512]);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_350: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_349, [-1, 2, 2, 7, 7, 512]);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_132: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_350, [0, 1, 3, 2, 4, 5]);  view_350 = None
        clone_142: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_351: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [-1, 14, 14, 512]);  clone_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_135: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_331, view_351);  view_331 = view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_352: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_135, [128, -1, 512]);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_28 = torch.ops.aten.var_mean.correction(view_352, [2], correction = 0, keepdim = True)
        getitem_95: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_28[0]
        getitem_96: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        sub_41: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_352, getitem_96);  getitem_96 = None
        add_136: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_95, 1e-05);  getitem_95 = None
        rsqrt_28: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_105: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = rsqrt_28 = None
        mul_106: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, arg193_1);  mul_105 = arg193_1 = None
        add_137: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, arg194_1);  mul_106 = arg194_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_353: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_137, [25088, 512]);  add_137 = None
        permute_133: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_50: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg196_1, view_353, permute_133);  arg196_1 = view_353 = permute_133 = None
        view_354: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [128, 196, 2048]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_107: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_354, 0.5)
        mul_108: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_354, 0.7071067811865476);  view_354 = None
        erf_12: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
        add_138: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_109: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, add_138);  mul_107 = add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_355: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_109, [25088, 2048]);  mul_109 = None
        permute_134: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_51: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg198_1, view_355, permute_134);  arg198_1 = view_355 = permute_134 = None
        view_356: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [128, 196, 512]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_139: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_352, view_356);  view_352 = view_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_357: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_139, [128, 14, 14, 512]);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_29 = torch.ops.aten.var_mean.correction(view_357, [3], correction = 0, keepdim = True)
        getitem_97: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_29[0]
        getitem_98: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        sub_42: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_357, getitem_98);  getitem_98 = None
        add_140: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_29: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_110: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_29);  sub_42 = rsqrt_29 = None
        mul_111: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, arg199_1);  mul_110 = arg199_1 = None
        add_141: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, arg200_1);  mul_111 = arg200_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_24: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_142: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_24, 3);  iota_24 = None
        fmod_24: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_142, 14);  add_142 = None
        index_37: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_141, [None, fmod_24]);  add_141 = fmod_24 = None
        iota_25: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_143: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_25, 3);  iota_25 = None
        fmod_25: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_143, 14);  add_143 = None
        index_38: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_37, [None, None, fmod_25]);  index_37 = fmod_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_358: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_38, [128, 2, 7, 2, 7, 512]);  index_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_135: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_358, [0, 1, 3, 2, 4, 5]);  view_358 = None
        clone_145: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_359: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [-1, 7, 7, 512]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_360: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_359, [-1, 49, 512]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_361: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_360, [25088, 512]);  view_360 = None
        permute_136: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_52: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg203_1, view_361, permute_136);  arg203_1 = view_361 = permute_136 = None
        view_362: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [512, 49, 1536]);  addmm_52 = None
        view_363: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_362, [512, 49, 3, 16, -1]);  view_362 = None
        permute_137: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_363, [2, 0, 3, 1, 4]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_13 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
        getitem_99: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_13[0]
        getitem_100: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_13[1]
        getitem_101: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_13[2];  unbind_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_112: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_99, 0.1767766952966369);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_52: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_112, [512, 16, 49, 32]);  mul_112 = None
        clone_146: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_364: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [8192, 49, 32]);  clone_146 = None
        permute_138: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_100, [0, 1, 3, 2]);  getitem_100 = None
        expand_53: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_138, [512, 16, 32, 49]);  permute_138 = None
        clone_147: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_365: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_147, [8192, 32, 49]);  clone_147 = None
        bmm_26: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_364, view_365);  view_364 = view_365 = None
        view_366: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [512, 16, 49, 49]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_367: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg205_1, [-1]);  arg205_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_39: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg204_1, [view_367]);  arg204_1 = view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_368: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_39, [49, 49, -1]);  index_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_139: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_368, [2, 0, 1]);  view_368 = None
        clone_148: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_25: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_148, 0);  clone_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_144: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_366, unsqueeze_25);  view_366 = unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_369: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_144, [-1, 4, 16, 49, 49]);  add_144 = None
        unsqueeze_26: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg201_1, 1);  arg201_1 = None
        unsqueeze_27: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 0);  unsqueeze_26 = None
        add_145: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_369, unsqueeze_27);  view_369 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_370: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_145, [-1, 16, 49, 49]);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_13: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_370, [-1], True)
        sub_43: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_370, amax_13);  view_370 = amax_13 = None
        exp_13: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_14: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_54: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_13, [512, 16, 49, 49]);  div_13 = None
        view_371: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_54, [8192, 49, 49]);  expand_54 = None
        expand_55: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_101, [512, 16, 49, 32]);  getitem_101 = None
        clone_150: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_372: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_150, [8192, 49, 32]);  clone_150 = None
        bmm_27: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_371, view_372);  view_371 = view_372 = None
        view_373: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [512, 16, 49, 32]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_140: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        clone_151: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_374: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [512, 49, 512]);  clone_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_375: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_374, [25088, 512]);  view_374 = None
        permute_141: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_53: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg207_1, view_375, permute_141);  arg207_1 = view_375 = permute_141 = None
        view_376: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [512, 49, 512]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_377: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_376, [-1, 7, 7, 512]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_378: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_377, [-1, 2, 2, 7, 7, 512]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_142: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_378, [0, 1, 3, 2, 4, 5]);  view_378 = None
        clone_153: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_379: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [-1, 14, 14, 512]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_26: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_146: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_26, 11);  iota_26 = None
        fmod_26: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_146, 14);  add_146 = None
        index_40: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_379, [None, fmod_26]);  view_379 = fmod_26 = None
        iota_27: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_147: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_27, 11);  iota_27 = None
        fmod_27: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_147, 14);  add_147 = None
        index_41: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_40, [None, None, fmod_27]);  index_40 = fmod_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_148: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_357, index_41);  view_357 = index_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_380: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_148, [128, -1, 512]);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_30 = torch.ops.aten.var_mean.correction(view_380, [2], correction = 0, keepdim = True)
        getitem_102: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_30[0]
        getitem_103: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        sub_44: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_380, getitem_103);  getitem_103 = None
        add_149: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_30: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_113: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = rsqrt_30 = None
        mul_114: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, arg208_1);  mul_113 = arg208_1 = None
        add_150: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, arg209_1);  mul_114 = arg209_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_381: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_150, [25088, 512]);  add_150 = None
        permute_143: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_54: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg211_1, view_381, permute_143);  arg211_1 = view_381 = permute_143 = None
        view_382: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [128, 196, 2048]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_115: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_382, 0.5)
        mul_116: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_382, 0.7071067811865476);  view_382 = None
        erf_13: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_151: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_117: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, add_151);  mul_115 = add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_383: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_117, [25088, 2048]);  mul_117 = None
        permute_144: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_55: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg213_1, view_383, permute_144);  arg213_1 = view_383 = permute_144 = None
        view_384: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [128, 196, 512]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_152: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_380, view_384);  view_380 = view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_385: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_152, [128, 14, 14, 512]);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_31 = torch.ops.aten.var_mean.correction(view_385, [3], correction = 0, keepdim = True)
        getitem_104: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_31[0]
        getitem_105: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        sub_45: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_385, getitem_105);  getitem_105 = None
        add_153: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_31: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_118: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_31);  sub_45 = rsqrt_31 = None
        mul_119: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, arg214_1);  mul_118 = arg214_1 = None
        add_154: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_119, arg215_1);  mul_119 = arg215_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_386: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_154, [128, 2, 7, 2, 7, 512]);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_145: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_386, [0, 1, 3, 2, 4, 5]);  view_386 = None
        clone_156: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        view_387: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_156, [-1, 7, 7, 512]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_388: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_387, [-1, 49, 512]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_389: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_388, [25088, 512]);  view_388 = None
        permute_146: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_56: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg217_1, view_389, permute_146);  arg217_1 = view_389 = permute_146 = None
        view_390: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [512, 49, 1536]);  addmm_56 = None
        view_391: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_390, [512, 49, 3, 16, -1]);  view_390 = None
        permute_147: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_391, [2, 0, 3, 1, 4]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_14 = torch.ops.aten.unbind.int(permute_147);  permute_147 = None
        getitem_106: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_14[0]
        getitem_107: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_14[1]
        getitem_108: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_14[2];  unbind_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_120: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_106, 0.1767766952966369);  getitem_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_56: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_120, [512, 16, 49, 32]);  mul_120 = None
        clone_157: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_392: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_157, [8192, 49, 32]);  clone_157 = None
        permute_148: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_107, [0, 1, 3, 2]);  getitem_107 = None
        expand_57: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_148, [512, 16, 32, 49]);  permute_148 = None
        clone_158: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_393: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_158, [8192, 32, 49]);  clone_158 = None
        bmm_28: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_392, view_393);  view_392 = view_393 = None
        view_394: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [512, 16, 49, 49]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_395: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg219_1, [-1]);  arg219_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_42: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg218_1, [view_395]);  arg218_1 = view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_396: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_42, [49, 49, -1]);  index_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_149: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_396, [2, 0, 1]);  view_396 = None
        clone_159: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_28: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_159, 0);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_155: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_394, unsqueeze_28);  view_394 = unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_14: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_155, [-1], True)
        sub_46: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_155, amax_14);  add_155 = amax_14 = None
        exp_14: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_15: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_58: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_14, [512, 16, 49, 49]);  div_14 = None
        view_397: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_58, [8192, 49, 49]);  expand_58 = None
        expand_59: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_108, [512, 16, 49, 32]);  getitem_108 = None
        clone_161: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_398: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [8192, 49, 32]);  clone_161 = None
        bmm_29: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_397, view_398);  view_397 = view_398 = None
        view_399: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [512, 16, 49, 32]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_150: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        clone_162: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_400: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [512, 49, 512]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_401: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_400, [25088, 512]);  view_400 = None
        permute_151: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_57: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg221_1, view_401, permute_151);  arg221_1 = view_401 = permute_151 = None
        view_402: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [512, 49, 512]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_403: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_402, [-1, 7, 7, 512]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_404: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_403, [-1, 2, 2, 7, 7, 512]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_152: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_404, [0, 1, 3, 2, 4, 5]);  view_404 = None
        clone_164: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        view_405: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_164, [-1, 14, 14, 512]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_156: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_385, view_405);  view_385 = view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_406: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_156, [128, -1, 512]);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_32 = torch.ops.aten.var_mean.correction(view_406, [2], correction = 0, keepdim = True)
        getitem_109: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_32[0]
        getitem_110: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        sub_47: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_406, getitem_110);  getitem_110 = None
        add_157: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_109, 1e-05);  getitem_109 = None
        rsqrt_32: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        mul_121: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = rsqrt_32 = None
        mul_122: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, arg222_1);  mul_121 = arg222_1 = None
        add_158: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, arg223_1);  mul_122 = arg223_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_407: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_158, [25088, 512]);  add_158 = None
        permute_153: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_58: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg225_1, view_407, permute_153);  arg225_1 = view_407 = permute_153 = None
        view_408: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [128, 196, 2048]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_123: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_408, 0.5)
        mul_124: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_408, 0.7071067811865476);  view_408 = None
        erf_14: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_124);  mul_124 = None
        add_159: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_125: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, add_159);  mul_123 = add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_409: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_125, [25088, 2048]);  mul_125 = None
        permute_154: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_59: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg227_1, view_409, permute_154);  arg227_1 = view_409 = permute_154 = None
        view_410: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [128, 196, 512]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_160: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_406, view_410);  view_406 = view_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_411: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_160, [128, 14, 14, 512]);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_33 = torch.ops.aten.var_mean.correction(view_411, [3], correction = 0, keepdim = True)
        getitem_111: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_33[0]
        getitem_112: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        sub_48: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_411, getitem_112);  getitem_112 = None
        add_161: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_111, 1e-05);  getitem_111 = None
        rsqrt_33: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        mul_126: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_33);  sub_48 = rsqrt_33 = None
        mul_127: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, arg228_1);  mul_126 = arg228_1 = None
        add_162: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, arg229_1);  mul_127 = arg229_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_28: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_163: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_28, 3);  iota_28 = None
        fmod_28: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_163, 14);  add_163 = None
        index_43: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_162, [None, fmod_28]);  add_162 = fmod_28 = None
        iota_29: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_164: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_29, 3);  iota_29 = None
        fmod_29: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_164, 14);  add_164 = None
        index_44: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_43, [None, None, fmod_29]);  index_43 = fmod_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_412: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_44, [128, 2, 7, 2, 7, 512]);  index_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_155: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 1, 3, 2, 4, 5]);  view_412 = None
        clone_167: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_413: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_167, [-1, 7, 7, 512]);  clone_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_414: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_413, [-1, 49, 512]);  view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_415: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_414, [25088, 512]);  view_414 = None
        permute_156: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_60: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg232_1, view_415, permute_156);  arg232_1 = view_415 = permute_156 = None
        view_416: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [512, 49, 1536]);  addmm_60 = None
        view_417: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_416, [512, 49, 3, 16, -1]);  view_416 = None
        permute_157: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_417, [2, 0, 3, 1, 4]);  view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_15 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
        getitem_113: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_15[0]
        getitem_114: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_15[1]
        getitem_115: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_15[2];  unbind_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_128: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_113, 0.1767766952966369);  getitem_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_60: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_128, [512, 16, 49, 32]);  mul_128 = None
        clone_168: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_418: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_168, [8192, 49, 32]);  clone_168 = None
        permute_158: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_114, [0, 1, 3, 2]);  getitem_114 = None
        expand_61: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_158, [512, 16, 32, 49]);  permute_158 = None
        clone_169: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_419: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [8192, 32, 49]);  clone_169 = None
        bmm_30: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_418, view_419);  view_418 = view_419 = None
        view_420: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [512, 16, 49, 49]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_421: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg234_1, [-1]);  arg234_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_45: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg233_1, [view_421]);  arg233_1 = view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_422: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_45, [49, 49, -1]);  index_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_159: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_422, [2, 0, 1]);  view_422 = None
        clone_170: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_29: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_170, 0);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_165: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_420, unsqueeze_29);  view_420 = unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_423: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_165, [-1, 4, 16, 49, 49]);  add_165 = None
        unsqueeze_30: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg230_1, 1);  arg230_1 = None
        unsqueeze_31: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 0);  unsqueeze_30 = None
        add_166: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_423, unsqueeze_31);  view_423 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_424: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_166, [-1, 16, 49, 49]);  add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_15: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_424, [-1], True)
        sub_49: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_424, amax_15);  view_424 = amax_15 = None
        exp_15: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_16: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_62: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_15, [512, 16, 49, 49]);  div_15 = None
        view_425: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_62, [8192, 49, 49]);  expand_62 = None
        expand_63: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_115, [512, 16, 49, 32]);  getitem_115 = None
        clone_172: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_426: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [8192, 49, 32]);  clone_172 = None
        bmm_31: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_425, view_426);  view_425 = view_426 = None
        view_427: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [512, 16, 49, 32]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_160: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        clone_173: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_428: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_173, [512, 49, 512]);  clone_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_429: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_428, [25088, 512]);  view_428 = None
        permute_161: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_61: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg236_1, view_429, permute_161);  arg236_1 = view_429 = permute_161 = None
        view_430: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [512, 49, 512]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_431: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_430, [-1, 7, 7, 512]);  view_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_432: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_431, [-1, 2, 2, 7, 7, 512]);  view_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_162: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 1, 3, 2, 4, 5]);  view_432 = None
        clone_175: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_433: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_175, [-1, 14, 14, 512]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_30: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_167: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_30, 11);  iota_30 = None
        fmod_30: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_167, 14);  add_167 = None
        index_46: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_433, [None, fmod_30]);  view_433 = fmod_30 = None
        iota_31: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_168: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_31, 11);  iota_31 = None
        fmod_31: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_168, 14);  add_168 = None
        index_47: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_46, [None, None, fmod_31]);  index_46 = fmod_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_169: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_411, index_47);  view_411 = index_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_434: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_169, [128, -1, 512]);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_34 = torch.ops.aten.var_mean.correction(view_434, [2], correction = 0, keepdim = True)
        getitem_116: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_34[0]
        getitem_117: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        sub_50: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_434, getitem_117);  getitem_117 = None
        add_170: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_34: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_129: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = rsqrt_34 = None
        mul_130: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, arg237_1);  mul_129 = arg237_1 = None
        add_171: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, arg238_1);  mul_130 = arg238_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_435: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_171, [25088, 512]);  add_171 = None
        permute_163: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_62: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg240_1, view_435, permute_163);  arg240_1 = view_435 = permute_163 = None
        view_436: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [128, 196, 2048]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_131: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_436, 0.5)
        mul_132: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_436, 0.7071067811865476);  view_436 = None
        erf_15: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_172: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_133: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, add_172);  mul_131 = add_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_437: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_133, [25088, 2048]);  mul_133 = None
        permute_164: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_63: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg242_1, view_437, permute_164);  arg242_1 = view_437 = permute_164 = None
        view_438: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [128, 196, 512]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_173: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_434, view_438);  view_434 = view_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_439: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_173, [128, 14, 14, 512]);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_35 = torch.ops.aten.var_mean.correction(view_439, [3], correction = 0, keepdim = True)
        getitem_118: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_35[0]
        getitem_119: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        sub_51: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_439, getitem_119);  getitem_119 = None
        add_174: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_35: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_134: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_35);  sub_51 = rsqrt_35 = None
        mul_135: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, arg243_1);  mul_134 = arg243_1 = None
        add_175: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_135, arg244_1);  mul_135 = arg244_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_440: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_175, [128, 2, 7, 2, 7, 512]);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_165: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_440, [0, 1, 3, 2, 4, 5]);  view_440 = None
        clone_178: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_441: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [-1, 7, 7, 512]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_442: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [-1, 49, 512]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_443: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_442, [25088, 512]);  view_442 = None
        permute_166: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_64: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg246_1, view_443, permute_166);  arg246_1 = view_443 = permute_166 = None
        view_444: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [512, 49, 1536]);  addmm_64 = None
        view_445: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_444, [512, 49, 3, 16, -1]);  view_444 = None
        permute_167: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_445, [2, 0, 3, 1, 4]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_16 = torch.ops.aten.unbind.int(permute_167);  permute_167 = None
        getitem_120: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_16[0]
        getitem_121: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_16[1]
        getitem_122: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_16[2];  unbind_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_136: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_120, 0.1767766952966369);  getitem_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_64: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_136, [512, 16, 49, 32]);  mul_136 = None
        clone_179: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_446: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [8192, 49, 32]);  clone_179 = None
        permute_168: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_121, [0, 1, 3, 2]);  getitem_121 = None
        expand_65: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_168, [512, 16, 32, 49]);  permute_168 = None
        clone_180: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_447: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_180, [8192, 32, 49]);  clone_180 = None
        bmm_32: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_446, view_447);  view_446 = view_447 = None
        view_448: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [512, 16, 49, 49]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_449: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg248_1, [-1]);  arg248_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_48: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg247_1, [view_449]);  arg247_1 = view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_450: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_48, [49, 49, -1]);  index_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_169: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_450, [2, 0, 1]);  view_450 = None
        clone_181: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_32: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_181, 0);  clone_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_176: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_448, unsqueeze_32);  view_448 = unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_16: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_176, [-1], True)
        sub_52: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_176, amax_16);  add_176 = amax_16 = None
        exp_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_17: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_66: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_16, [512, 16, 49, 49]);  div_16 = None
        view_451: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_66, [8192, 49, 49]);  expand_66 = None
        expand_67: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_122, [512, 16, 49, 32]);  getitem_122 = None
        clone_183: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_452: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_183, [8192, 49, 32]);  clone_183 = None
        bmm_33: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_451, view_452);  view_451 = view_452 = None
        view_453: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [512, 16, 49, 32]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_170: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        clone_184: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_454: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_184, [512, 49, 512]);  clone_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_455: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_454, [25088, 512]);  view_454 = None
        permute_171: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_65: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg250_1, view_455, permute_171);  arg250_1 = view_455 = permute_171 = None
        view_456: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [512, 49, 512]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_457: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_456, [-1, 7, 7, 512]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_458: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_457, [-1, 2, 2, 7, 7, 512]);  view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_172: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2, 4, 5]);  view_458 = None
        clone_186: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_459: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [-1, 14, 14, 512]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_177: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_439, view_459);  view_439 = view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_460: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_177, [128, -1, 512]);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_36 = torch.ops.aten.var_mean.correction(view_460, [2], correction = 0, keepdim = True)
        getitem_123: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_36[0]
        getitem_124: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        sub_53: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_460, getitem_124);  getitem_124 = None
        add_178: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_123, 1e-05);  getitem_123 = None
        rsqrt_36: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_137: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = rsqrt_36 = None
        mul_138: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, arg251_1);  mul_137 = arg251_1 = None
        add_179: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_138, arg252_1);  mul_138 = arg252_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_461: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_179, [25088, 512]);  add_179 = None
        permute_173: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_66: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg254_1, view_461, permute_173);  arg254_1 = view_461 = permute_173 = None
        view_462: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [128, 196, 2048]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_139: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_462, 0.5)
        mul_140: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_462, 0.7071067811865476);  view_462 = None
        erf_16: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_180: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_141: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, add_180);  mul_139 = add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_463: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_141, [25088, 2048]);  mul_141 = None
        permute_174: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_67: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg256_1, view_463, permute_174);  arg256_1 = view_463 = permute_174 = None
        view_464: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [128, 196, 512]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_181: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_460, view_464);  view_460 = view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_465: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_181, [128, 14, 14, 512]);  add_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_37 = torch.ops.aten.var_mean.correction(view_465, [3], correction = 0, keepdim = True)
        getitem_125: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_37[0]
        getitem_126: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        sub_54: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_465, getitem_126);  getitem_126 = None
        add_182: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_125, 1e-05);  getitem_125 = None
        rsqrt_37: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        mul_142: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_37);  sub_54 = rsqrt_37 = None
        mul_143: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_142, arg257_1);  mul_142 = arg257_1 = None
        add_183: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_143, arg258_1);  mul_143 = arg258_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_32: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_184: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_32, 3);  iota_32 = None
        fmod_32: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_184, 14);  add_184 = None
        index_49: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_183, [None, fmod_32]);  add_183 = fmod_32 = None
        iota_33: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_185: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_33, 3);  iota_33 = None
        fmod_33: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_185, 14);  add_185 = None
        index_50: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_49, [None, None, fmod_33]);  index_49 = fmod_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_466: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_50, [128, 2, 7, 2, 7, 512]);  index_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_175: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 1, 3, 2, 4, 5]);  view_466 = None
        clone_189: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_467: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [-1, 7, 7, 512]);  clone_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_468: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_467, [-1, 49, 512]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_469: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_468, [25088, 512]);  view_468 = None
        permute_176: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        addmm_68: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg261_1, view_469, permute_176);  arg261_1 = view_469 = permute_176 = None
        view_470: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [512, 49, 1536]);  addmm_68 = None
        view_471: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_470, [512, 49, 3, 16, -1]);  view_470 = None
        permute_177: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_471, [2, 0, 3, 1, 4]);  view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_17 = torch.ops.aten.unbind.int(permute_177);  permute_177 = None
        getitem_127: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_17[0]
        getitem_128: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_17[1]
        getitem_129: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_17[2];  unbind_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_144: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_127, 0.1767766952966369);  getitem_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_68: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_144, [512, 16, 49, 32]);  mul_144 = None
        clone_190: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_472: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_190, [8192, 49, 32]);  clone_190 = None
        permute_178: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_128, [0, 1, 3, 2]);  getitem_128 = None
        expand_69: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_178, [512, 16, 32, 49]);  permute_178 = None
        clone_191: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_473: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_191, [8192, 32, 49]);  clone_191 = None
        bmm_34: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_472, view_473);  view_472 = view_473 = None
        view_474: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [512, 16, 49, 49]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_475: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg263_1, [-1]);  arg263_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_51: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg262_1, [view_475]);  arg262_1 = view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_476: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_51, [49, 49, -1]);  index_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_179: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_476, [2, 0, 1]);  view_476 = None
        clone_192: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_33: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_192, 0);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_186: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_474, unsqueeze_33);  view_474 = unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_477: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_186, [-1, 4, 16, 49, 49]);  add_186 = None
        unsqueeze_34: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg259_1, 1);  arg259_1 = None
        unsqueeze_35: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 0);  unsqueeze_34 = None
        add_187: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_477, unsqueeze_35);  view_477 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_478: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_187, [-1, 16, 49, 49]);  add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_17: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_478, [-1], True)
        sub_55: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_478, amax_17);  view_478 = amax_17 = None
        exp_17: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_18: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_70: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_17, [512, 16, 49, 49]);  div_17 = None
        view_479: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_70, [8192, 49, 49]);  expand_70 = None
        expand_71: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_129, [512, 16, 49, 32]);  getitem_129 = None
        clone_194: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_480: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_194, [8192, 49, 32]);  clone_194 = None
        bmm_35: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_479, view_480);  view_479 = view_480 = None
        view_481: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [512, 16, 49, 32]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_180: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
        clone_195: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_482: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_195, [512, 49, 512]);  clone_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_483: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_482, [25088, 512]);  view_482 = None
        permute_181: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_69: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg265_1, view_483, permute_181);  arg265_1 = view_483 = permute_181 = None
        view_484: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [512, 49, 512]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_485: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_484, [-1, 7, 7, 512]);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_486: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [-1, 2, 2, 7, 7, 512]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_182: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 1, 3, 2, 4, 5]);  view_486 = None
        clone_197: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_487: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_197, [-1, 14, 14, 512]);  clone_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_34: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_188: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_34, 11);  iota_34 = None
        fmod_34: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_188, 14);  add_188 = None
        index_52: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_487, [None, fmod_34]);  view_487 = fmod_34 = None
        iota_35: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_189: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_35, 11);  iota_35 = None
        fmod_35: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_189, 14);  add_189 = None
        index_53: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_52, [None, None, fmod_35]);  index_52 = fmod_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_190: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_465, index_53);  view_465 = index_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_488: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_190, [128, -1, 512]);  add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_38 = torch.ops.aten.var_mean.correction(view_488, [2], correction = 0, keepdim = True)
        getitem_130: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_38[0]
        getitem_131: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        sub_56: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_488, getitem_131);  getitem_131 = None
        add_191: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_38: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        mul_145: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = rsqrt_38 = None
        mul_146: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, arg266_1);  mul_145 = arg266_1 = None
        add_192: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, arg267_1);  mul_146 = arg267_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_489: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_192, [25088, 512]);  add_192 = None
        permute_183: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_70: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg269_1, view_489, permute_183);  arg269_1 = view_489 = permute_183 = None
        view_490: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [128, 196, 2048]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_147: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_490, 0.5)
        mul_148: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_490, 0.7071067811865476);  view_490 = None
        erf_17: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_193: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_149: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, add_193);  mul_147 = add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_491: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_149, [25088, 2048]);  mul_149 = None
        permute_184: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        addmm_71: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg271_1, view_491, permute_184);  arg271_1 = view_491 = permute_184 = None
        view_492: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [128, 196, 512]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_194: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_488, view_492);  view_488 = view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_493: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_194, [128, 14, 14, 512]);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_39 = torch.ops.aten.var_mean.correction(view_493, [3], correction = 0, keepdim = True)
        getitem_132: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_39[0]
        getitem_133: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        sub_57: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_493, getitem_133);  getitem_133 = None
        add_195: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_39: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        mul_150: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_39);  sub_57 = rsqrt_39 = None
        mul_151: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, arg272_1);  mul_150 = arg272_1 = None
        add_196: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, arg273_1);  mul_151 = arg273_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_494: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_196, [128, 2, 7, 2, 7, 512]);  add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_185: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_494, [0, 1, 3, 2, 4, 5]);  view_494 = None
        clone_200: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_495: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_200, [-1, 7, 7, 512]);  clone_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_496: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_495, [-1, 49, 512]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_497: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_496, [25088, 512]);  view_496 = None
        permute_186: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_72: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg275_1, view_497, permute_186);  arg275_1 = view_497 = permute_186 = None
        view_498: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [512, 49, 1536]);  addmm_72 = None
        view_499: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_498, [512, 49, 3, 16, -1]);  view_498 = None
        permute_187: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_499, [2, 0, 3, 1, 4]);  view_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_18 = torch.ops.aten.unbind.int(permute_187);  permute_187 = None
        getitem_134: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_18[0]
        getitem_135: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_18[1]
        getitem_136: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_18[2];  unbind_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_152: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_134, 0.1767766952966369);  getitem_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_72: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_152, [512, 16, 49, 32]);  mul_152 = None
        clone_201: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_500: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [8192, 49, 32]);  clone_201 = None
        permute_188: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_135, [0, 1, 3, 2]);  getitem_135 = None
        expand_73: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_188, [512, 16, 32, 49]);  permute_188 = None
        clone_202: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_501: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_202, [8192, 32, 49]);  clone_202 = None
        bmm_36: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_500, view_501);  view_500 = view_501 = None
        view_502: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [512, 16, 49, 49]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_503: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg277_1, [-1]);  arg277_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_54: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg276_1, [view_503]);  arg276_1 = view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_504: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_54, [49, 49, -1]);  index_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_189: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_504, [2, 0, 1]);  view_504 = None
        clone_203: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_36: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_203, 0);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_197: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_502, unsqueeze_36);  view_502 = unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_18: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_197, [-1], True)
        sub_58: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_197, amax_18);  add_197 = amax_18 = None
        exp_18: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_19: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_74: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_18, [512, 16, 49, 49]);  div_18 = None
        view_505: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_74, [8192, 49, 49]);  expand_74 = None
        expand_75: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_136, [512, 16, 49, 32]);  getitem_136 = None
        clone_205: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_506: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_205, [8192, 49, 32]);  clone_205 = None
        bmm_37: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_505, view_506);  view_505 = view_506 = None
        view_507: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [512, 16, 49, 32]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_190: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
        clone_206: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_508: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_206, [512, 49, 512]);  clone_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_509: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_508, [25088, 512]);  view_508 = None
        permute_191: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_73: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg279_1, view_509, permute_191);  arg279_1 = view_509 = permute_191 = None
        view_510: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [512, 49, 512]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_511: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_510, [-1, 7, 7, 512]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_512: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [-1, 2, 2, 7, 7, 512]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_192: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 1, 3, 2, 4, 5]);  view_512 = None
        clone_208: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_513: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_208, [-1, 14, 14, 512]);  clone_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_198: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_493, view_513);  view_493 = view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_514: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_198, [128, -1, 512]);  add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_40 = torch.ops.aten.var_mean.correction(view_514, [2], correction = 0, keepdim = True)
        getitem_137: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_40[0]
        getitem_138: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        sub_59: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_514, getitem_138);  getitem_138 = None
        add_199: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_137, 1e-05);  getitem_137 = None
        rsqrt_40: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        mul_153: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = rsqrt_40 = None
        mul_154: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_153, arg280_1);  mul_153 = arg280_1 = None
        add_200: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_154, arg281_1);  mul_154 = arg281_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_515: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_200, [25088, 512]);  add_200 = None
        permute_193: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_74: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg283_1, view_515, permute_193);  arg283_1 = view_515 = permute_193 = None
        view_516: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [128, 196, 2048]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_155: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_516, 0.5)
        mul_156: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_516, 0.7071067811865476);  view_516 = None
        erf_18: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_156);  mul_156 = None
        add_201: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_157: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, add_201);  mul_155 = add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_517: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_157, [25088, 2048]);  mul_157 = None
        permute_194: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_75: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg285_1, view_517, permute_194);  arg285_1 = view_517 = permute_194 = None
        view_518: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [128, 196, 512]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_202: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_514, view_518);  view_514 = view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_519: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_202, [128, 14, 14, 512]);  add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_41 = torch.ops.aten.var_mean.correction(view_519, [3], correction = 0, keepdim = True)
        getitem_139: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_41[0]
        getitem_140: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        sub_60: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_519, getitem_140);  getitem_140 = None
        add_203: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_139, 1e-05);  getitem_139 = None
        rsqrt_41: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_158: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_41);  sub_60 = rsqrt_41 = None
        mul_159: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, arg286_1);  mul_158 = arg286_1 = None
        add_204: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_159, arg287_1);  mul_159 = arg287_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_36: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_205: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_36, 3);  iota_36 = None
        fmod_36: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_205, 14);  add_205 = None
        index_55: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_204, [None, fmod_36]);  add_204 = fmod_36 = None
        iota_37: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_206: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_37, 3);  iota_37 = None
        fmod_37: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_206, 14);  add_206 = None
        index_56: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_55, [None, None, fmod_37]);  index_55 = fmod_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_520: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_56, [128, 2, 7, 2, 7, 512]);  index_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_195: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_520, [0, 1, 3, 2, 4, 5]);  view_520 = None
        clone_211: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_521: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_211, [-1, 7, 7, 512]);  clone_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_522: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_521, [-1, 49, 512]);  view_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_523: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_522, [25088, 512]);  view_522 = None
        permute_196: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_76: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg290_1, view_523, permute_196);  arg290_1 = view_523 = permute_196 = None
        view_524: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [512, 49, 1536]);  addmm_76 = None
        view_525: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_524, [512, 49, 3, 16, -1]);  view_524 = None
        permute_197: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_525, [2, 0, 3, 1, 4]);  view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_19 = torch.ops.aten.unbind.int(permute_197);  permute_197 = None
        getitem_141: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_19[0]
        getitem_142: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_19[1]
        getitem_143: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_19[2];  unbind_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_160: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_141, 0.1767766952966369);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_76: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_160, [512, 16, 49, 32]);  mul_160 = None
        clone_212: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_526: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_212, [8192, 49, 32]);  clone_212 = None
        permute_198: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_142, [0, 1, 3, 2]);  getitem_142 = None
        expand_77: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_198, [512, 16, 32, 49]);  permute_198 = None
        clone_213: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_527: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_213, [8192, 32, 49]);  clone_213 = None
        bmm_38: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_526, view_527);  view_526 = view_527 = None
        view_528: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [512, 16, 49, 49]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_529: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg292_1, [-1]);  arg292_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_57: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg291_1, [view_529]);  arg291_1 = view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_530: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_57, [49, 49, -1]);  index_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_199: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_530, [2, 0, 1]);  view_530 = None
        clone_214: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_37: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_214, 0);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_207: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_528, unsqueeze_37);  view_528 = unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_531: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_207, [-1, 4, 16, 49, 49]);  add_207 = None
        unsqueeze_38: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg288_1, 1);  arg288_1 = None
        unsqueeze_39: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 0);  unsqueeze_38 = None
        add_208: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_531, unsqueeze_39);  view_531 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_532: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_208, [-1, 16, 49, 49]);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_19: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_532, [-1], True)
        sub_61: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_532, amax_19);  view_532 = amax_19 = None
        exp_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_20: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_78: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_19, [512, 16, 49, 49]);  div_19 = None
        view_533: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_78, [8192, 49, 49]);  expand_78 = None
        expand_79: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_143, [512, 16, 49, 32]);  getitem_143 = None
        clone_216: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_534: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_216, [8192, 49, 32]);  clone_216 = None
        bmm_39: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_533, view_534);  view_533 = view_534 = None
        view_535: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [512, 16, 49, 32]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_200: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        clone_217: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_536: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_217, [512, 49, 512]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_537: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_536, [25088, 512]);  view_536 = None
        permute_201: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_77: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg294_1, view_537, permute_201);  arg294_1 = view_537 = permute_201 = None
        view_538: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [512, 49, 512]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_539: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_538, [-1, 7, 7, 512]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_540: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_539, [-1, 2, 2, 7, 7, 512]);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_202: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_540, [0, 1, 3, 2, 4, 5]);  view_540 = None
        clone_219: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_541: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_219, [-1, 14, 14, 512]);  clone_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_38: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_209: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_38, 11);  iota_38 = None
        fmod_38: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_209, 14);  add_209 = None
        index_58: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_541, [None, fmod_38]);  view_541 = fmod_38 = None
        iota_39: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_210: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_39, 11);  iota_39 = None
        fmod_39: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_210, 14);  add_210 = None
        index_59: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_58, [None, None, fmod_39]);  index_58 = fmod_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_211: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_519, index_59);  view_519 = index_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_542: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_211, [128, -1, 512]);  add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_42 = torch.ops.aten.var_mean.correction(view_542, [2], correction = 0, keepdim = True)
        getitem_144: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_42[0]
        getitem_145: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        sub_62: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_542, getitem_145);  getitem_145 = None
        add_212: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_42: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        mul_161: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = rsqrt_42 = None
        mul_162: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, arg295_1);  mul_161 = arg295_1 = None
        add_213: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, arg296_1);  mul_162 = arg296_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_543: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_213, [25088, 512]);  add_213 = None
        permute_203: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_78: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg298_1, view_543, permute_203);  arg298_1 = view_543 = permute_203 = None
        view_544: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [128, 196, 2048]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_163: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_544, 0.5)
        mul_164: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_544, 0.7071067811865476);  view_544 = None
        erf_19: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
        add_214: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_165: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, add_214);  mul_163 = add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_545: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_165, [25088, 2048]);  mul_165 = None
        permute_204: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_79: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg300_1, view_545, permute_204);  arg300_1 = view_545 = permute_204 = None
        view_546: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [128, 196, 512]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_215: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_542, view_546);  view_542 = view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_547: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_215, [128, 14, 14, 512]);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_43 = torch.ops.aten.var_mean.correction(view_547, [3], correction = 0, keepdim = True)
        getitem_146: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_43[0]
        getitem_147: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        sub_63: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_547, getitem_147);  getitem_147 = None
        add_216: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_43: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        mul_166: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_43);  sub_63 = rsqrt_43 = None
        mul_167: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, arg301_1);  mul_166 = arg301_1 = None
        add_217: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_167, arg302_1);  mul_167 = arg302_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_548: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_217, [128, 2, 7, 2, 7, 512]);  add_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_205: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_548, [0, 1, 3, 2, 4, 5]);  view_548 = None
        clone_222: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_549: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_222, [-1, 7, 7, 512]);  clone_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_550: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_549, [-1, 49, 512]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_551: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_550, [25088, 512]);  view_550 = None
        permute_206: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_80: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg304_1, view_551, permute_206);  arg304_1 = view_551 = permute_206 = None
        view_552: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [512, 49, 1536]);  addmm_80 = None
        view_553: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_552, [512, 49, 3, 16, -1]);  view_552 = None
        permute_207: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_553, [2, 0, 3, 1, 4]);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_20 = torch.ops.aten.unbind.int(permute_207);  permute_207 = None
        getitem_148: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_20[0]
        getitem_149: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_20[1]
        getitem_150: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_20[2];  unbind_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_168: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_148, 0.1767766952966369);  getitem_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_80: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_168, [512, 16, 49, 32]);  mul_168 = None
        clone_223: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_554: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_223, [8192, 49, 32]);  clone_223 = None
        permute_208: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_149, [0, 1, 3, 2]);  getitem_149 = None
        expand_81: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_208, [512, 16, 32, 49]);  permute_208 = None
        clone_224: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_555: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_224, [8192, 32, 49]);  clone_224 = None
        bmm_40: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_554, view_555);  view_554 = view_555 = None
        view_556: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [512, 16, 49, 49]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_557: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg306_1, [-1]);  arg306_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_60: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg305_1, [view_557]);  arg305_1 = view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_558: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_60, [49, 49, -1]);  index_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_209: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_558, [2, 0, 1]);  view_558 = None
        clone_225: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_40: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_225, 0);  clone_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_218: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_556, unsqueeze_40);  view_556 = unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_20: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_218, [-1], True)
        sub_64: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_218, amax_20);  add_218 = amax_20 = None
        exp_20: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_21: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_82: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_20, [512, 16, 49, 49]);  div_20 = None
        view_559: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_82, [8192, 49, 49]);  expand_82 = None
        expand_83: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_150, [512, 16, 49, 32]);  getitem_150 = None
        clone_227: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_560: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_227, [8192, 49, 32]);  clone_227 = None
        bmm_41: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_559, view_560);  view_559 = view_560 = None
        view_561: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [512, 16, 49, 32]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_210: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
        clone_228: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_562: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_228, [512, 49, 512]);  clone_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_563: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_562, [25088, 512]);  view_562 = None
        permute_211: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_81: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg308_1, view_563, permute_211);  arg308_1 = view_563 = permute_211 = None
        view_564: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [512, 49, 512]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_565: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_564, [-1, 7, 7, 512]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_566: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_565, [-1, 2, 2, 7, 7, 512]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_212: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_566, [0, 1, 3, 2, 4, 5]);  view_566 = None
        clone_230: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_567: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_230, [-1, 14, 14, 512]);  clone_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_219: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_547, view_567);  view_547 = view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_568: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_219, [128, -1, 512]);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_44 = torch.ops.aten.var_mean.correction(view_568, [2], correction = 0, keepdim = True)
        getitem_151: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_44[0]
        getitem_152: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        sub_65: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_568, getitem_152);  getitem_152 = None
        add_220: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_151, 1e-05);  getitem_151 = None
        rsqrt_44: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        mul_169: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = rsqrt_44 = None
        mul_170: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, arg309_1);  mul_169 = arg309_1 = None
        add_221: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, arg310_1);  mul_170 = arg310_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_569: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_221, [25088, 512]);  add_221 = None
        permute_213: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_82: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg312_1, view_569, permute_213);  arg312_1 = view_569 = permute_213 = None
        view_570: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [128, 196, 2048]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_171: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_570, 0.5)
        mul_172: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_570, 0.7071067811865476);  view_570 = None
        erf_20: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
        add_222: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_173: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, add_222);  mul_171 = add_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_571: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_173, [25088, 2048]);  mul_173 = None
        permute_214: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_83: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg314_1, view_571, permute_214);  arg314_1 = view_571 = permute_214 = None
        view_572: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [128, 196, 512]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_223: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_568, view_572);  view_568 = view_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_573: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_223, [128, 14, 14, 512]);  add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_45 = torch.ops.aten.var_mean.correction(view_573, [3], correction = 0, keepdim = True)
        getitem_153: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_45[0]
        getitem_154: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        sub_66: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_573, getitem_154);  getitem_154 = None
        add_224: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_153, 1e-05);  getitem_153 = None
        rsqrt_45: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        mul_174: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_45);  sub_66 = rsqrt_45 = None
        mul_175: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, arg315_1);  mul_174 = arg315_1 = None
        add_225: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_175, arg316_1);  mul_175 = arg316_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_40: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_226: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_40, 3);  iota_40 = None
        fmod_40: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_226, 14);  add_226 = None
        index_61: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_225, [None, fmod_40]);  add_225 = fmod_40 = None
        iota_41: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_227: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_41, 3);  iota_41 = None
        fmod_41: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_227, 14);  add_227 = None
        index_62: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_61, [None, None, fmod_41]);  index_61 = fmod_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_574: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_62, [128, 2, 7, 2, 7, 512]);  index_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_215: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_574, [0, 1, 3, 2, 4, 5]);  view_574 = None
        clone_233: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_575: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_233, [-1, 7, 7, 512]);  clone_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_576: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_575, [-1, 49, 512]);  view_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_577: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_576, [25088, 512]);  view_576 = None
        permute_216: "f32[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        addmm_84: "f32[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg319_1, view_577, permute_216);  arg319_1 = view_577 = permute_216 = None
        view_578: "f32[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [512, 49, 1536]);  addmm_84 = None
        view_579: "f32[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_578, [512, 49, 3, 16, -1]);  view_578 = None
        permute_217: "f32[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_579, [2, 0, 3, 1, 4]);  view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_21 = torch.ops.aten.unbind.int(permute_217);  permute_217 = None
        getitem_155: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_21[0]
        getitem_156: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_21[1]
        getitem_157: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_21[2];  unbind_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_176: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_155, 0.1767766952966369);  getitem_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_84: "f32[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_176, [512, 16, 49, 32]);  mul_176 = None
        clone_234: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_580: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_234, [8192, 49, 32]);  clone_234 = None
        permute_218: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_156, [0, 1, 3, 2]);  getitem_156 = None
        expand_85: "f32[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_218, [512, 16, 32, 49]);  permute_218 = None
        clone_235: "f32[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_581: "f32[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_235, [8192, 32, 49]);  clone_235 = None
        bmm_42: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_580, view_581);  view_580 = view_581 = None
        view_582: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [512, 16, 49, 49]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_583: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg321_1, [-1]);  arg321_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_63: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg320_1, [view_583]);  arg320_1 = view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_584: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_63, [49, 49, -1]);  index_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_219: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_584, [2, 0, 1]);  view_584 = None
        clone_236: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_41: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_236, 0);  clone_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_228: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_582, unsqueeze_41);  view_582 = unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_585: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_228, [-1, 4, 16, 49, 49]);  add_228 = None
        unsqueeze_42: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg317_1, 1);  arg317_1 = None
        unsqueeze_43: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 0);  unsqueeze_42 = None
        add_229: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_585, unsqueeze_43);  view_585 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_586: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_229, [-1, 16, 49, 49]);  add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_21: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_586, [-1], True)
        sub_67: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_586, amax_21);  view_586 = amax_21 = None
        exp_21: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_22: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_86: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_21, [512, 16, 49, 49]);  div_21 = None
        view_587: "f32[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_86, [8192, 49, 49]);  expand_86 = None
        expand_87: "f32[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_157, [512, 16, 49, 32]);  getitem_157 = None
        clone_238: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_588: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_238, [8192, 49, 32]);  clone_238 = None
        bmm_43: "f32[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_587, view_588);  view_587 = view_588 = None
        view_589: "f32[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [512, 16, 49, 32]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_220: "f32[512, 49, 16, 32][25088, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
        clone_239: "f32[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_590: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_239, [512, 49, 512]);  clone_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_591: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_590, [25088, 512]);  view_590 = None
        permute_221: "f32[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        addmm_85: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg323_1, view_591, permute_221);  arg323_1 = view_591 = permute_221 = None
        view_592: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_85, [512, 49, 512]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_593: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_592, [-1, 7, 7, 512]);  view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_594: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [-1, 2, 2, 7, 7, 512]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_222: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_594, [0, 1, 3, 2, 4, 5]);  view_594 = None
        clone_241: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_595: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_241, [-1, 14, 14, 512]);  clone_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_42: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_230: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_42, 11);  iota_42 = None
        fmod_42: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_230, 14);  add_230 = None
        index_64: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_595, [None, fmod_42]);  view_595 = fmod_42 = None
        iota_43: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_231: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_43, 11);  iota_43 = None
        fmod_43: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_231, 14);  add_231 = None
        index_65: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_64, [None, None, fmod_43]);  index_64 = fmod_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_232: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_573, index_65);  view_573 = index_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_596: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_232, [128, -1, 512]);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_46 = torch.ops.aten.var_mean.correction(view_596, [2], correction = 0, keepdim = True)
        getitem_158: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_46[0]
        getitem_159: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        sub_68: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_596, getitem_159);  getitem_159 = None
        add_233: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_46: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        mul_177: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_46);  sub_68 = rsqrt_46 = None
        mul_178: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, arg324_1);  mul_177 = arg324_1 = None
        add_234: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_178, arg325_1);  mul_178 = arg325_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_597: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_234, [25088, 512]);  add_234 = None
        permute_223: "f32[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_86: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg327_1, view_597, permute_223);  arg327_1 = view_597 = permute_223 = None
        view_598: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [128, 196, 2048]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_179: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_598, 0.5)
        mul_180: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_598, 0.7071067811865476);  view_598 = None
        erf_21: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
        add_235: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_181: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, add_235);  mul_179 = add_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_599: "f32[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_181, [25088, 2048]);  mul_181 = None
        permute_224: "f32[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_87: "f32[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg329_1, view_599, permute_224);  arg329_1 = view_599 = permute_224 = None
        view_600: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_87, [128, 196, 512]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_236: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_596, view_600);  view_596 = view_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_601: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_236, [128, 14, 14, 512]);  add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_602: "f32[128, 7, 2, 7, 2, 512][100352, 14336, 7168, 1024, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_601, [128, 7, 2, 7, 2, 512]);  view_601 = None
        permute_225: "f32[128, 7, 7, 2, 2, 512][100352, 14336, 1024, 512, 7168, 1]cuda:0" = torch.ops.aten.permute.default(view_602, [0, 1, 3, 4, 2, 5]);  view_602 = None
        clone_244: "f32[128, 7, 7, 2, 2, 512][100352, 14336, 2048, 1024, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_603: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_244, [128, 7, 7, 2048]);  clone_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_47 = torch.ops.aten.var_mean.correction(view_603, [3], correction = 0, keepdim = True)
        getitem_160: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_47[0]
        getitem_161: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        sub_69: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_603, getitem_161);  view_603 = getitem_161 = None
        add_237: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_47: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        mul_182: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_47);  sub_69 = rsqrt_47 = None
        mul_183: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, arg330_1);  mul_182 = arg330_1 = None
        add_238: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_183, arg331_1);  mul_183 = arg331_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_604: "f32[6272, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_238, [6272, 2048]);  add_238 = None
        permute_226: "f32[2048, 1024][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        mm_2: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_604, permute_226);  view_604 = permute_226 = None
        view_605: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [128, 7, 7, 1024]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_48 = torch.ops.aten.var_mean.correction(view_605, [3], correction = 0, keepdim = True)
        getitem_162: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_48[0]
        getitem_163: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        sub_70: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_605, getitem_163);  getitem_163 = None
        add_239: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_48: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_184: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_48);  sub_70 = rsqrt_48 = None
        mul_185: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, arg333_1);  mul_184 = arg333_1 = None
        add_240: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_185, arg334_1);  mul_185 = arg334_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_606: "f32[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_240, [128, 1, 7, 1, 7, 1024]);  add_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_227: "f32[128, 1, 1, 7, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_606, [0, 1, 3, 2, 4, 5]);  view_606 = None
        view_607: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_227, [-1, 7, 7, 1024]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_608: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_607, [-1, 49, 1024]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_609: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_608, [6272, 1024]);  view_608 = None
        permute_228: "f32[1024, 3072][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_88: "f32[6272, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg336_1, view_609, permute_228);  arg336_1 = view_609 = permute_228 = None
        view_610: "f32[128, 49, 3072][150528, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [128, 49, 3072]);  addmm_88 = None
        view_611: "f32[128, 49, 3, 32, 32][150528, 3072, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_610, [128, 49, 3, 32, -1]);  view_610 = None
        permute_229: "f32[3, 128, 32, 49, 32][1024, 150528, 32, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_611, [2, 0, 3, 1, 4]);  view_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_22 = torch.ops.aten.unbind.int(permute_229);  permute_229 = None
        getitem_164: "f32[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_22[0]
        getitem_165: "f32[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_22[1]
        getitem_166: "f32[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_22[2];  unbind_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_186: "f32[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_164, 0.1767766952966369);  getitem_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_88: "f32[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_186, [128, 32, 49, 32]);  mul_186 = None
        clone_245: "f32[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_612: "f32[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_245, [4096, 49, 32]);  clone_245 = None
        permute_230: "f32[128, 32, 32, 49][150528, 32, 1, 3072]cuda:0" = torch.ops.aten.permute.default(getitem_165, [0, 1, 3, 2]);  getitem_165 = None
        expand_89: "f32[128, 32, 32, 49][150528, 32, 1, 3072]cuda:0" = torch.ops.aten.expand.default(permute_230, [128, 32, 32, 49]);  permute_230 = None
        clone_246: "f32[128, 32, 32, 49][50176, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_613: "f32[4096, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_246, [4096, 32, 49]);  clone_246 = None
        bmm_44: "f32[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_612, view_613);  view_612 = view_613 = None
        view_614: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [128, 32, 49, 49]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_615: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg338_1, [-1]);  arg338_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_66: "f32[2401, 32][32, 1]cuda:0" = torch.ops.aten.index.Tensor(arg337_1, [view_615]);  arg337_1 = view_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_616: "f32[49, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(index_66, [49, 49, -1]);  index_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_231: "f32[32, 49, 49][1, 1568, 32]cuda:0" = torch.ops.aten.permute.default(view_616, [2, 0, 1]);  view_616 = None
        clone_247: "f32[32, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_44: "f32[1, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_247, 0);  clone_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_241: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_614, unsqueeze_44);  view_614 = unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_22: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_241, [-1], True)
        sub_71: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_241, amax_22);  add_241 = amax_22 = None
        exp_22: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        sum_23: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_90: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_22, [128, 32, 49, 49]);  div_22 = None
        view_617: "f32[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_90, [4096, 49, 49]);  expand_90 = None
        expand_91: "f32[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = torch.ops.aten.expand.default(getitem_166, [128, 32, 49, 32]);  getitem_166 = None
        clone_249: "f32[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_618: "f32[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_249, [4096, 49, 32]);  clone_249 = None
        bmm_45: "f32[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_617, view_618);  view_617 = view_618 = None
        view_619: "f32[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [128, 32, 49, 32]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_232: "f32[128, 49, 32, 32][50176, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None
        clone_250: "f32[128, 49, 32, 32][50176, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_620: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_250, [128, 49, 1024]);  clone_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_621: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_620, [6272, 1024]);  view_620 = None
        permute_233: "f32[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_89: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg340_1, view_621, permute_233);  arg340_1 = view_621 = permute_233 = None
        view_622: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_89, [128, 49, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_623: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_622, [-1, 7, 7, 1024]);  view_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_624: "f32[128, 1, 1, 7, 7, 1024][50176, 50176, 50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_623, [-1, 1, 1, 7, 7, 1024]);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_234: "f32[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 50176, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_624, [0, 1, 3, 2, 4, 5]);  view_624 = None
        view_625: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_234, [-1, 7, 7, 1024]);  permute_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_242: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_605, view_625);  view_605 = view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_626: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_242, [128, -1, 1024]);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_49 = torch.ops.aten.var_mean.correction(view_626, [2], correction = 0, keepdim = True)
        getitem_167: "f32[128, 49, 1][49, 1, 1]cuda:0" = var_mean_49[0]
        getitem_168: "f32[128, 49, 1][49, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        sub_72: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_626, getitem_168);  getitem_168 = None
        add_243: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_167, 1e-05);  getitem_167 = None
        rsqrt_49: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_187: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_49);  sub_72 = rsqrt_49 = None
        mul_188: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, arg341_1);  mul_187 = arg341_1 = None
        add_244: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, arg342_1);  mul_188 = arg342_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_627: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_244, [6272, 1024]);  add_244 = None
        permute_235: "f32[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_90: "f32[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg344_1, view_627, permute_235);  arg344_1 = view_627 = permute_235 = None
        view_628: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [128, 49, 4096]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_189: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_628, 0.5)
        mul_190: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_628, 0.7071067811865476);  view_628 = None
        erf_22: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_190);  mul_190 = None
        add_245: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_191: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, add_245);  mul_189 = add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_629: "f32[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_191, [6272, 4096]);  mul_191 = None
        permute_236: "f32[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_91: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg346_1, view_629, permute_236);  arg346_1 = view_629 = permute_236 = None
        view_630: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_91, [128, 49, 1024]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_246: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_626, view_630);  view_626 = view_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_631: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_246, [128, 7, 7, 1024]);  add_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_50 = torch.ops.aten.var_mean.correction(view_631, [3], correction = 0, keepdim = True)
        getitem_169: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_50[0]
        getitem_170: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        sub_73: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_631, getitem_170);  getitem_170 = None
        add_247: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_169, 1e-05);  getitem_169 = None
        rsqrt_50: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_192: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_50);  sub_73 = rsqrt_50 = None
        mul_193: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, arg347_1);  mul_192 = arg347_1 = None
        add_248: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, arg348_1);  mul_193 = arg348_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_632: "f32[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_248, [128, 1, 7, 1, 7, 1024]);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_237: "f32[128, 1, 1, 7, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_632, [0, 1, 3, 2, 4, 5]);  view_632 = None
        view_633: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_237, [-1, 7, 7, 1024]);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_634: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_633, [-1, 49, 1024]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_635: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_634, [6272, 1024]);  view_634 = None
        permute_238: "f32[1024, 3072][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_92: "f32[6272, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg350_1, view_635, permute_238);  arg350_1 = view_635 = permute_238 = None
        view_636: "f32[128, 49, 3072][150528, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_92, [128, 49, 3072]);  addmm_92 = None
        view_637: "f32[128, 49, 3, 32, 32][150528, 3072, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_636, [128, 49, 3, 32, -1]);  view_636 = None
        permute_239: "f32[3, 128, 32, 49, 32][1024, 150528, 32, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_637, [2, 0, 3, 1, 4]);  view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_23 = torch.ops.aten.unbind.int(permute_239);  permute_239 = None
        getitem_171: "f32[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_23[0]
        getitem_172: "f32[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_23[1]
        getitem_173: "f32[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_23[2];  unbind_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_194: "f32[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_171, 0.1767766952966369);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_92: "f32[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_194, [128, 32, 49, 32]);  mul_194 = None
        clone_254: "f32[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_638: "f32[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_254, [4096, 49, 32]);  clone_254 = None
        permute_240: "f32[128, 32, 32, 49][150528, 32, 1, 3072]cuda:0" = torch.ops.aten.permute.default(getitem_172, [0, 1, 3, 2]);  getitem_172 = None
        expand_93: "f32[128, 32, 32, 49][150528, 32, 1, 3072]cuda:0" = torch.ops.aten.expand.default(permute_240, [128, 32, 32, 49]);  permute_240 = None
        clone_255: "f32[128, 32, 32, 49][50176, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_639: "f32[4096, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_255, [4096, 32, 49]);  clone_255 = None
        bmm_46: "f32[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_638, view_639);  view_638 = view_639 = None
        view_640: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [128, 32, 49, 49]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_641: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(arg352_1, [-1]);  arg352_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_67: "f32[2401, 32][32, 1]cuda:0" = torch.ops.aten.index.Tensor(arg351_1, [view_641]);  arg351_1 = view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_642: "f32[49, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(index_67, [49, 49, -1]);  index_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_241: "f32[32, 49, 49][1, 1568, 32]cuda:0" = torch.ops.aten.permute.default(view_642, [2, 0, 1]);  view_642 = None
        clone_256: "f32[32, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_45: "f32[1, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_256, 0);  clone_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_249: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_640, unsqueeze_45);  view_640 = unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_23: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_249, [-1], True)
        sub_74: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_249, amax_23);  add_249 = amax_23 = None
        exp_23: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
        sum_24: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_94: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(div_23, [128, 32, 49, 49]);  div_23 = None
        view_643: "f32[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_94, [4096, 49, 49]);  expand_94 = None
        expand_95: "f32[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = torch.ops.aten.expand.default(getitem_173, [128, 32, 49, 32]);  getitem_173 = None
        clone_258: "f32[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_644: "f32[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_258, [4096, 49, 32]);  clone_258 = None
        bmm_47: "f32[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_643, view_644);  view_643 = view_644 = None
        view_645: "f32[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [128, 32, 49, 32]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_242: "f32[128, 49, 32, 32][50176, 32, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
        clone_259: "f32[128, 49, 32, 32][50176, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_646: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_259, [128, 49, 1024]);  clone_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_647: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_646, [6272, 1024]);  view_646 = None
        permute_243: "f32[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_93: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg354_1, view_647, permute_243);  arg354_1 = view_647 = permute_243 = None
        view_648: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_93, [128, 49, 1024]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_649: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_648, [-1, 7, 7, 1024]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_650: "f32[128, 1, 1, 7, 7, 1024][50176, 50176, 50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_649, [-1, 1, 1, 7, 7, 1024]);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_244: "f32[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 50176, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_650, [0, 1, 3, 2, 4, 5]);  view_650 = None
        view_651: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_244, [-1, 7, 7, 1024]);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_250: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_631, view_651);  view_631 = view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_652: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_250, [128, -1, 1024]);  add_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_51 = torch.ops.aten.var_mean.correction(view_652, [2], correction = 0, keepdim = True)
        getitem_174: "f32[128, 49, 1][49, 1, 1]cuda:0" = var_mean_51[0]
        getitem_175: "f32[128, 49, 1][49, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        sub_75: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_652, getitem_175);  getitem_175 = None
        add_251: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_51: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        mul_195: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_51);  sub_75 = rsqrt_51 = None
        mul_196: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_195, arg355_1);  mul_195 = arg355_1 = None
        add_252: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_196, arg356_1);  mul_196 = arg356_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_653: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_252, [6272, 1024]);  add_252 = None
        permute_245: "f32[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_94: "f32[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg358_1, view_653, permute_245);  arg358_1 = view_653 = permute_245 = None
        view_654: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [128, 49, 4096]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_197: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_654, 0.5)
        mul_198: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_654, 0.7071067811865476);  view_654 = None
        erf_23: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_198);  mul_198 = None
        add_253: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_199: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_197, add_253);  mul_197 = add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_655: "f32[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_199, [6272, 4096]);  mul_199 = None
        permute_246: "f32[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_95: "f32[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg360_1, view_655, permute_246);  arg360_1 = view_655 = permute_246 = None
        view_656: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_95, [128, 49, 1024]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_254: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_652, view_656);  view_652 = view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_657: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_254, [128, 7, 7, 1024]);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        var_mean_52 = torch.ops.aten.var_mean.correction(view_657, [3], correction = 0, keepdim = True)
        getitem_176: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_52[0]
        getitem_177: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        sub_76: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_657, getitem_177);  view_657 = getitem_177 = None
        add_255: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_52: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        mul_200: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_52);  sub_76 = rsqrt_52 = None
        mul_201: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, arg361_1);  mul_200 = arg361_1 = None
        add_256: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_201, arg362_1);  mul_201 = arg362_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:65 in forward, code: return x.mean(self.dim, keepdim=not self.flatten)
        mean: "f32[128, 1024][1024, 1]cuda:0" = torch.ops.aten.mean.dim(add_256, [1, 2]);  add_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_247: "f32[1024, 1000][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_96: "f32[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg364_1, mean, permute_247);  arg364_1 = mean = permute_247 = None
        return (addmm_96,)

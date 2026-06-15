class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 128][128, 1]cuda:0", arg1_1: "bf16[50257, 2048][2048, 1]cuda:0", arg2_1: "bf16[2048, 2048][2048, 1]cuda:0", arg3_1: "bf16[2048][1]cuda:0", arg4_1: "bf16[2048][1]cuda:0", arg5_1: "bf16[2048, 2048][2048, 1]cuda:0", arg6_1: "bf16[2048, 2048][2048, 1]cuda:0", arg7_1: "bf16[2048, 2048][2048, 1]cuda:0", arg8_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg9_1: "bf16[2048, 2048][2048, 1]cuda:0", arg10_1: "bf16[2048][1]cuda:0", arg11_1: "bf16[2048][1]cuda:0", arg12_1: "bf16[2048][1]cuda:0", arg13_1: "bf16[8192, 2048][2048, 1]cuda:0", arg14_1: "bf16[8192][1]cuda:0", arg15_1: "bf16[2048, 8192][8192, 1]cuda:0", arg16_1: "bf16[2048][1]cuda:0", arg17_1: "bf16[2048][1]cuda:0", arg18_1: "bf16[2048][1]cuda:0", arg19_1: "bf16[2048, 2048][2048, 1]cuda:0", arg20_1: "bf16[2048, 2048][2048, 1]cuda:0", arg21_1: "bf16[2048, 2048][2048, 1]cuda:0", arg22_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg23_1: "bf16[2048, 2048][2048, 1]cuda:0", arg24_1: "bf16[2048][1]cuda:0", arg25_1: "bf16[2048][1]cuda:0", arg26_1: "bf16[2048][1]cuda:0", arg27_1: "bf16[8192, 2048][2048, 1]cuda:0", arg28_1: "bf16[8192][1]cuda:0", arg29_1: "bf16[2048, 8192][8192, 1]cuda:0", arg30_1: "bf16[2048][1]cuda:0", arg31_1: "bf16[2048][1]cuda:0", arg32_1: "bf16[2048][1]cuda:0", arg33_1: "bf16[2048, 2048][2048, 1]cuda:0", arg34_1: "bf16[2048, 2048][2048, 1]cuda:0", arg35_1: "bf16[2048, 2048][2048, 1]cuda:0", arg36_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg37_1: "bf16[2048, 2048][2048, 1]cuda:0", arg38_1: "bf16[2048][1]cuda:0", arg39_1: "bf16[2048][1]cuda:0", arg40_1: "bf16[2048][1]cuda:0", arg41_1: "bf16[8192, 2048][2048, 1]cuda:0", arg42_1: "bf16[8192][1]cuda:0", arg43_1: "bf16[2048, 8192][8192, 1]cuda:0", arg44_1: "bf16[2048][1]cuda:0", arg45_1: "bf16[2048][1]cuda:0", arg46_1: "bf16[2048][1]cuda:0", arg47_1: "bf16[2048, 2048][2048, 1]cuda:0", arg48_1: "bf16[2048, 2048][2048, 1]cuda:0", arg49_1: "bf16[2048, 2048][2048, 1]cuda:0", arg50_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg51_1: "bf16[2048, 2048][2048, 1]cuda:0", arg52_1: "bf16[2048][1]cuda:0", arg53_1: "bf16[2048][1]cuda:0", arg54_1: "bf16[2048][1]cuda:0", arg55_1: "bf16[8192, 2048][2048, 1]cuda:0", arg56_1: "bf16[8192][1]cuda:0", arg57_1: "bf16[2048, 8192][8192, 1]cuda:0", arg58_1: "bf16[2048][1]cuda:0", arg59_1: "bf16[2048][1]cuda:0", arg60_1: "bf16[2048][1]cuda:0", arg61_1: "bf16[2048, 2048][2048, 1]cuda:0", arg62_1: "bf16[2048, 2048][2048, 1]cuda:0", arg63_1: "bf16[2048, 2048][2048, 1]cuda:0", arg64_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg65_1: "bf16[2048, 2048][2048, 1]cuda:0", arg66_1: "bf16[2048][1]cuda:0", arg67_1: "bf16[2048][1]cuda:0", arg68_1: "bf16[2048][1]cuda:0", arg69_1: "bf16[8192, 2048][2048, 1]cuda:0", arg70_1: "bf16[8192][1]cuda:0", arg71_1: "bf16[2048, 8192][8192, 1]cuda:0", arg72_1: "bf16[2048][1]cuda:0", arg73_1: "bf16[2048][1]cuda:0", arg74_1: "bf16[2048][1]cuda:0", arg75_1: "bf16[2048, 2048][2048, 1]cuda:0", arg76_1: "bf16[2048, 2048][2048, 1]cuda:0", arg77_1: "bf16[2048, 2048][2048, 1]cuda:0", arg78_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg79_1: "bf16[2048, 2048][2048, 1]cuda:0", arg80_1: "bf16[2048][1]cuda:0", arg81_1: "bf16[2048][1]cuda:0", arg82_1: "bf16[2048][1]cuda:0", arg83_1: "bf16[8192, 2048][2048, 1]cuda:0", arg84_1: "bf16[8192][1]cuda:0", arg85_1: "bf16[2048, 8192][8192, 1]cuda:0", arg86_1: "bf16[2048][1]cuda:0", arg87_1: "bf16[2048][1]cuda:0", arg88_1: "bf16[2048][1]cuda:0", arg89_1: "bf16[2048, 2048][2048, 1]cuda:0", arg90_1: "bf16[2048, 2048][2048, 1]cuda:0", arg91_1: "bf16[2048, 2048][2048, 1]cuda:0", arg92_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg93_1: "bf16[2048, 2048][2048, 1]cuda:0", arg94_1: "bf16[2048][1]cuda:0", arg95_1: "bf16[2048][1]cuda:0", arg96_1: "bf16[2048][1]cuda:0", arg97_1: "bf16[8192, 2048][2048, 1]cuda:0", arg98_1: "bf16[8192][1]cuda:0", arg99_1: "bf16[2048, 8192][8192, 1]cuda:0", arg100_1: "bf16[2048][1]cuda:0", arg101_1: "bf16[2048][1]cuda:0", arg102_1: "bf16[2048][1]cuda:0", arg103_1: "bf16[2048, 2048][2048, 1]cuda:0", arg104_1: "bf16[2048, 2048][2048, 1]cuda:0", arg105_1: "bf16[2048, 2048][2048, 1]cuda:0", arg106_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg107_1: "bf16[2048, 2048][2048, 1]cuda:0", arg108_1: "bf16[2048][1]cuda:0", arg109_1: "bf16[2048][1]cuda:0", arg110_1: "bf16[2048][1]cuda:0", arg111_1: "bf16[8192, 2048][2048, 1]cuda:0", arg112_1: "bf16[8192][1]cuda:0", arg113_1: "bf16[2048, 8192][8192, 1]cuda:0", arg114_1: "bf16[2048][1]cuda:0", arg115_1: "bf16[2048][1]cuda:0", arg116_1: "bf16[2048][1]cuda:0", arg117_1: "bf16[2048, 2048][2048, 1]cuda:0", arg118_1: "bf16[2048, 2048][2048, 1]cuda:0", arg119_1: "bf16[2048, 2048][2048, 1]cuda:0", arg120_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg121_1: "bf16[2048, 2048][2048, 1]cuda:0", arg122_1: "bf16[2048][1]cuda:0", arg123_1: "bf16[2048][1]cuda:0", arg124_1: "bf16[2048][1]cuda:0", arg125_1: "bf16[8192, 2048][2048, 1]cuda:0", arg126_1: "bf16[8192][1]cuda:0", arg127_1: "bf16[2048, 8192][8192, 1]cuda:0", arg128_1: "bf16[2048][1]cuda:0", arg129_1: "bf16[2048][1]cuda:0", arg130_1: "bf16[2048][1]cuda:0", arg131_1: "bf16[2048, 2048][2048, 1]cuda:0", arg132_1: "bf16[2048, 2048][2048, 1]cuda:0", arg133_1: "bf16[2048, 2048][2048, 1]cuda:0", arg134_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg135_1: "bf16[2048, 2048][2048, 1]cuda:0", arg136_1: "bf16[2048][1]cuda:0", arg137_1: "bf16[2048][1]cuda:0", arg138_1: "bf16[2048][1]cuda:0", arg139_1: "bf16[8192, 2048][2048, 1]cuda:0", arg140_1: "bf16[8192][1]cuda:0", arg141_1: "bf16[2048, 8192][8192, 1]cuda:0", arg142_1: "bf16[2048][1]cuda:0", arg143_1: "bf16[2048][1]cuda:0", arg144_1: "bf16[2048][1]cuda:0", arg145_1: "bf16[2048, 2048][2048, 1]cuda:0", arg146_1: "bf16[2048, 2048][2048, 1]cuda:0", arg147_1: "bf16[2048, 2048][2048, 1]cuda:0", arg148_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg149_1: "bf16[2048, 2048][2048, 1]cuda:0", arg150_1: "bf16[2048][1]cuda:0", arg151_1: "bf16[2048][1]cuda:0", arg152_1: "bf16[2048][1]cuda:0", arg153_1: "bf16[8192, 2048][2048, 1]cuda:0", arg154_1: "bf16[8192][1]cuda:0", arg155_1: "bf16[2048, 8192][8192, 1]cuda:0", arg156_1: "bf16[2048][1]cuda:0", arg157_1: "bf16[2048][1]cuda:0", arg158_1: "bf16[2048][1]cuda:0", arg159_1: "bf16[2048, 2048][2048, 1]cuda:0", arg160_1: "bf16[2048, 2048][2048, 1]cuda:0", arg161_1: "bf16[2048, 2048][2048, 1]cuda:0", arg162_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg163_1: "bf16[2048, 2048][2048, 1]cuda:0", arg164_1: "bf16[2048][1]cuda:0", arg165_1: "bf16[2048][1]cuda:0", arg166_1: "bf16[2048][1]cuda:0", arg167_1: "bf16[8192, 2048][2048, 1]cuda:0", arg168_1: "bf16[8192][1]cuda:0", arg169_1: "bf16[2048, 8192][8192, 1]cuda:0", arg170_1: "bf16[2048][1]cuda:0", arg171_1: "bf16[2048][1]cuda:0", arg172_1: "bf16[2048][1]cuda:0", arg173_1: "bf16[2048, 2048][2048, 1]cuda:0", arg174_1: "bf16[2048, 2048][2048, 1]cuda:0", arg175_1: "bf16[2048, 2048][2048, 1]cuda:0", arg176_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg177_1: "bf16[2048, 2048][2048, 1]cuda:0", arg178_1: "bf16[2048][1]cuda:0", arg179_1: "bf16[2048][1]cuda:0", arg180_1: "bf16[2048][1]cuda:0", arg181_1: "bf16[8192, 2048][2048, 1]cuda:0", arg182_1: "bf16[8192][1]cuda:0", arg183_1: "bf16[2048, 8192][8192, 1]cuda:0", arg184_1: "bf16[2048][1]cuda:0", arg185_1: "bf16[2048][1]cuda:0", arg186_1: "bf16[2048][1]cuda:0", arg187_1: "bf16[2048, 2048][2048, 1]cuda:0", arg188_1: "bf16[2048, 2048][2048, 1]cuda:0", arg189_1: "bf16[2048, 2048][2048, 1]cuda:0", arg190_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg191_1: "bf16[2048, 2048][2048, 1]cuda:0", arg192_1: "bf16[2048][1]cuda:0", arg193_1: "bf16[2048][1]cuda:0", arg194_1: "bf16[2048][1]cuda:0", arg195_1: "bf16[8192, 2048][2048, 1]cuda:0", arg196_1: "bf16[8192][1]cuda:0", arg197_1: "bf16[2048, 8192][8192, 1]cuda:0", arg198_1: "bf16[2048][1]cuda:0", arg199_1: "bf16[2048][1]cuda:0", arg200_1: "bf16[2048][1]cuda:0", arg201_1: "bf16[2048, 2048][2048, 1]cuda:0", arg202_1: "bf16[2048, 2048][2048, 1]cuda:0", arg203_1: "bf16[2048, 2048][2048, 1]cuda:0", arg204_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg205_1: "bf16[2048, 2048][2048, 1]cuda:0", arg206_1: "bf16[2048][1]cuda:0", arg207_1: "bf16[2048][1]cuda:0", arg208_1: "bf16[2048][1]cuda:0", arg209_1: "bf16[8192, 2048][2048, 1]cuda:0", arg210_1: "bf16[8192][1]cuda:0", arg211_1: "bf16[2048, 8192][8192, 1]cuda:0", arg212_1: "bf16[2048][1]cuda:0", arg213_1: "bf16[2048][1]cuda:0", arg214_1: "bf16[2048][1]cuda:0", arg215_1: "bf16[2048, 2048][2048, 1]cuda:0", arg216_1: "bf16[2048, 2048][2048, 1]cuda:0", arg217_1: "bf16[2048, 2048][2048, 1]cuda:0", arg218_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg219_1: "bf16[2048, 2048][2048, 1]cuda:0", arg220_1: "bf16[2048][1]cuda:0", arg221_1: "bf16[2048][1]cuda:0", arg222_1: "bf16[2048][1]cuda:0", arg223_1: "bf16[8192, 2048][2048, 1]cuda:0", arg224_1: "bf16[8192][1]cuda:0", arg225_1: "bf16[2048, 8192][8192, 1]cuda:0", arg226_1: "bf16[2048][1]cuda:0", arg227_1: "bf16[2048][1]cuda:0", arg228_1: "bf16[2048][1]cuda:0", arg229_1: "bf16[2048, 2048][2048, 1]cuda:0", arg230_1: "bf16[2048, 2048][2048, 1]cuda:0", arg231_1: "bf16[2048, 2048][2048, 1]cuda:0", arg232_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg233_1: "bf16[2048, 2048][2048, 1]cuda:0", arg234_1: "bf16[2048][1]cuda:0", arg235_1: "bf16[2048][1]cuda:0", arg236_1: "bf16[2048][1]cuda:0", arg237_1: "bf16[8192, 2048][2048, 1]cuda:0", arg238_1: "bf16[8192][1]cuda:0", arg239_1: "bf16[2048, 8192][8192, 1]cuda:0", arg240_1: "bf16[2048][1]cuda:0", arg241_1: "bf16[2048][1]cuda:0", arg242_1: "bf16[2048][1]cuda:0", arg243_1: "bf16[2048, 2048][2048, 1]cuda:0", arg244_1: "bf16[2048, 2048][2048, 1]cuda:0", arg245_1: "bf16[2048, 2048][2048, 1]cuda:0", arg246_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg247_1: "bf16[2048, 2048][2048, 1]cuda:0", arg248_1: "bf16[2048][1]cuda:0", arg249_1: "bf16[2048][1]cuda:0", arg250_1: "bf16[2048][1]cuda:0", arg251_1: "bf16[8192, 2048][2048, 1]cuda:0", arg252_1: "bf16[8192][1]cuda:0", arg253_1: "bf16[2048, 8192][8192, 1]cuda:0", arg254_1: "bf16[2048][1]cuda:0", arg255_1: "bf16[2048][1]cuda:0", arg256_1: "bf16[2048][1]cuda:0", arg257_1: "bf16[2048, 2048][2048, 1]cuda:0", arg258_1: "bf16[2048, 2048][2048, 1]cuda:0", arg259_1: "bf16[2048, 2048][2048, 1]cuda:0", arg260_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg261_1: "bf16[2048, 2048][2048, 1]cuda:0", arg262_1: "bf16[2048][1]cuda:0", arg263_1: "bf16[2048][1]cuda:0", arg264_1: "bf16[2048][1]cuda:0", arg265_1: "bf16[8192, 2048][2048, 1]cuda:0", arg266_1: "bf16[8192][1]cuda:0", arg267_1: "bf16[2048, 8192][8192, 1]cuda:0", arg268_1: "bf16[2048][1]cuda:0", arg269_1: "bf16[2048][1]cuda:0", arg270_1: "bf16[2048][1]cuda:0", arg271_1: "bf16[2048, 2048][2048, 1]cuda:0", arg272_1: "bf16[2048, 2048][2048, 1]cuda:0", arg273_1: "bf16[2048, 2048][2048, 1]cuda:0", arg274_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg275_1: "bf16[2048, 2048][2048, 1]cuda:0", arg276_1: "bf16[2048][1]cuda:0", arg277_1: "bf16[2048][1]cuda:0", arg278_1: "bf16[2048][1]cuda:0", arg279_1: "bf16[8192, 2048][2048, 1]cuda:0", arg280_1: "bf16[8192][1]cuda:0", arg281_1: "bf16[2048, 8192][8192, 1]cuda:0", arg282_1: "bf16[2048][1]cuda:0", arg283_1: "bf16[2048][1]cuda:0", arg284_1: "bf16[2048][1]cuda:0", arg285_1: "bf16[2048, 2048][2048, 1]cuda:0", arg286_1: "bf16[2048, 2048][2048, 1]cuda:0", arg287_1: "bf16[2048, 2048][2048, 1]cuda:0", arg288_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg289_1: "bf16[2048, 2048][2048, 1]cuda:0", arg290_1: "bf16[2048][1]cuda:0", arg291_1: "bf16[2048][1]cuda:0", arg292_1: "bf16[2048][1]cuda:0", arg293_1: "bf16[8192, 2048][2048, 1]cuda:0", arg294_1: "bf16[8192][1]cuda:0", arg295_1: "bf16[2048, 8192][8192, 1]cuda:0", arg296_1: "bf16[2048][1]cuda:0", arg297_1: "bf16[2048][1]cuda:0", arg298_1: "bf16[2048][1]cuda:0", arg299_1: "bf16[2048, 2048][2048, 1]cuda:0", arg300_1: "bf16[2048, 2048][2048, 1]cuda:0", arg301_1: "bf16[2048, 2048][2048, 1]cuda:0", arg302_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg303_1: "bf16[2048, 2048][2048, 1]cuda:0", arg304_1: "bf16[2048][1]cuda:0", arg305_1: "bf16[2048][1]cuda:0", arg306_1: "bf16[2048][1]cuda:0", arg307_1: "bf16[8192, 2048][2048, 1]cuda:0", arg308_1: "bf16[8192][1]cuda:0", arg309_1: "bf16[2048, 8192][8192, 1]cuda:0", arg310_1: "bf16[2048][1]cuda:0", arg311_1: "bf16[2048][1]cuda:0", arg312_1: "bf16[2048][1]cuda:0", arg313_1: "bf16[2048, 2048][2048, 1]cuda:0", arg314_1: "bf16[2048, 2048][2048, 1]cuda:0", arg315_1: "bf16[2048, 2048][2048, 1]cuda:0", arg316_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg317_1: "bf16[2048, 2048][2048, 1]cuda:0", arg318_1: "bf16[2048][1]cuda:0", arg319_1: "bf16[2048][1]cuda:0", arg320_1: "bf16[2048][1]cuda:0", arg321_1: "bf16[8192, 2048][2048, 1]cuda:0", arg322_1: "bf16[8192][1]cuda:0", arg323_1: "bf16[2048, 8192][8192, 1]cuda:0", arg324_1: "bf16[2048][1]cuda:0", arg325_1: "bf16[2048][1]cuda:0", arg326_1: "bf16[2048][1]cuda:0", arg327_1: "bf16[2048, 2048][2048, 1]cuda:0", arg328_1: "bf16[2048, 2048][2048, 1]cuda:0", arg329_1: "bf16[2048, 2048][2048, 1]cuda:0", arg330_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", arg331_1: "bf16[2048, 2048][2048, 1]cuda:0", arg332_1: "bf16[2048][1]cuda:0", arg333_1: "bf16[2048][1]cuda:0", arg334_1: "bf16[2048][1]cuda:0", arg335_1: "bf16[8192, 2048][2048, 1]cuda:0", arg336_1: "bf16[8192][1]cuda:0", arg337_1: "bf16[2048, 8192][8192, 1]cuda:0", arg338_1: "bf16[2048][1]cuda:0", arg339_1: "bf16[2048][1]cuda:0", arg340_1: "bf16[2048][1]cuda:0", arg341_1: "i64[32, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "bf16[][]cuda:0" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:444 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:451 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:452 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:462 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "bf16[1, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        add_3: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant1: "f32[][]cuda:0" = self._tensor_constant1;  _tensor_constant1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_4: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg8_1, 2, 0, 128);  arg8_1 = None
        slice_5: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 128);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_2: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_1: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_5: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type_1: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 2048])
        permute: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 128, 2048]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_6: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [32, 128, 16, 128]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_3: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_8: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_2: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_8, [32, 16, 128, 128]);  convert_element_type_8 = None
        clone_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_9: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [512, 128, 128]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_2: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 2048])
        permute_1: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_1: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
        view_3: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [32, 128, 2048]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_7: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [32, 128, 16, 128]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_4: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_9: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_6: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_9, [0, 1, 3, 2]);  convert_element_type_9 = None
        expand_3: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_6, [32, 16, 128, 128]);  permute_6 = None
        clone_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_10: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [512, 128, 128]);  clone_2 = None
        bmm: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 16, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_2: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_5, view_11, full_default_2);  slice_5 = view_11 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_7: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_8: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_4: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_5: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[32, 128][0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze, [32, -1]);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[32, 1][0, 1]cuda:0" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[32, 1][1, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[32, 129][129, 1]cuda:0" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_3: "i64[32, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 1, 129)
        slice_2: "i64[32, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 0, 128);  cat = None
        sub_1: "i64[32, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[32, 128][128, 1]cuda:0" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[32][1]cuda:0" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[32, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[32, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_1, [32, -1, 128, 128]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default, full_default_1);  expand_1 = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_1, where);  where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_6, [-1], True)
        sub_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_1: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_10: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_4: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_10, [32, 16, 128, 128]);  convert_element_type_10 = None
        view_12: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_4, [512, 128, 128]);  expand_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_4: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 2048]);  convert_element_type_1 = None
        permute_2: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_2: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 128, 2048]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_8: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [32, 128, 16, 128]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_5: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_5: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [32, 16, 128, 128]);  permute_5 = None
        clone_4: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_13: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [512, 128, 128]);  clone_4 = None
        bmm_1: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
        view_14: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [32, 16, 128, 128]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_7: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_5: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_15: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [32, 128, 2048]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_16: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [4096, 2048]);  view_15 = None
        permute_8: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg10_1, view_16, permute_8);  arg10_1 = view_16 = permute_8 = None
        view_17: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [32, 128, 2048]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_7: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_17, add_3);  view_17 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_16: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_4: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_3);  convert_element_type_16 = getitem_3 = None
        add_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_2: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_3: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, arg11_1);  mul_2 = arg11_1 = None
        add_9: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, arg12_1);  mul_3 = arg12_1 = None
        convert_element_type_17: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_18: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_17, [4096, 2048]);  convert_element_type_17 = None
        permute_9: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg14_1, view_18, permute_9);  arg14_1 = view_18 = permute_9 = None
        view_19: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 128, 8192]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        pow_1: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
        mul_5: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_10: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_19, mul_5);  view_19 = mul_5 = None
        mul_6: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None
        tanh: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_11: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, add_11);  mul_4 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_20: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_7, [4096, 8192]);  mul_7 = None
        permute_10: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_2: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, view_20, permute_10);  arg16_1 = view_20 = permute_10 = None
        view_21: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 128, 2048]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_12: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_7, view_21);  add_7 = view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_24: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant2: "f32[][]cuda:0" = self._tensor_constant2;  _tensor_constant2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_6: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg22_1, 2, 0, 128);  arg22_1 = None
        slice_7: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 128);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_5: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_5);  convert_element_type_24 = getitem_5 = None
        add_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_8: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = rsqrt_2 = None
        mul_9: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, arg17_1);  mul_8 = arg17_1 = None
        add_14: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, arg18_1);  mul_9 = arg18_1 = None
        convert_element_type_25: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_22: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [4096, 2048])
        permute_11: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_3: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
        view_23: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [32, 128, 2048]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_28: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [32, 128, 16, 128]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_14: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_32: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_6: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_32, [32, 16, 128, 128]);  convert_element_type_32 = None
        clone_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_31: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [512, 128, 128]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_24: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [4096, 2048])
        permute_12: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_4: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
        view_25: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [32, 128, 2048]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_29: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [32, 128, 16, 128]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_15: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_33: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_15, torch.float32);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_17: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_33, [0, 1, 3, 2]);  convert_element_type_33 = None
        expand_7: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_17, [32, 16, 128, 128]);  permute_17 = None
        clone_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_32: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [512, 128, 128]);  clone_9 = None
        bmm_2: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
        view_33: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [32, 16, 128, 128]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_3: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_7, view_33, full_default_3);  slice_7 = view_33 = full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_2, where);  where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_1: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_15, [-1], True)
        sub_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_15, amax_1);  add_15 = amax_1 = None
        exp_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_34: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_8: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_34, [32, 16, 128, 128]);  convert_element_type_34 = None
        view_34: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_8, [512, 128, 128]);  expand_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_26: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [4096, 2048]);  convert_element_type_25 = None
        permute_13: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_5: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_26, permute_13);  view_26 = permute_13 = None
        view_27: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [32, 128, 2048]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_30: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_27, [32, 128, 16, 128]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_16: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_9: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_16, [32, 16, 128, 128]);  permute_16 = None
        clone_11: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_35: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [512, 128, 128]);  clone_11 = None
        bmm_3: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [32, 16, 128, 128]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_18: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_12: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_37: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [32, 128, 2048]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_38: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [4096, 2048]);  view_37 = None
        permute_19: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_3: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg24_1, view_38, permute_19);  arg24_1 = view_38 = permute_19 = None
        view_39: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [32, 128, 2048]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_16: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_39, add_12);  view_39 = add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_40: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_7: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_7);  convert_element_type_40 = getitem_7 = None
        add_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_10: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_3);  sub_7 = rsqrt_3 = None
        mul_11: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, arg25_1);  mul_10 = arg25_1 = None
        add_18: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, arg26_1);  mul_11 = arg26_1 = None
        convert_element_type_41: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_40: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_41, [4096, 2048]);  convert_element_type_41 = None
        permute_20: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_4: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg28_1, view_40, permute_20);  arg28_1 = view_40 = permute_20 = None
        view_41: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 8192]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        pow_2: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_41, 3.0)
        mul_13: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_19: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_41, mul_13);  view_41 = mul_13 = None
        mul_14: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_19, 0.7978845608028654);  add_19 = None
        tanh_1: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_20: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_20);  mul_12 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_42: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_15, [4096, 8192]);  mul_15 = None
        permute_21: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_5: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg30_1, view_42, permute_21);  arg30_1 = view_42 = permute_21 = None
        view_43: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 128, 2048]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_21: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_16, view_43);  add_16 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_48: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant3: "f32[][]cuda:0" = self._tensor_constant3;  _tensor_constant3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_8: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg36_1, 2, 0, 128);  arg36_1 = None
        slice_9: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 128);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_8: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_9);  convert_element_type_48 = getitem_9 = None
        add_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_16: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = rsqrt_4 = None
        mul_17: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, arg31_1);  mul_16 = arg31_1 = None
        add_23: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, arg32_1);  mul_17 = arg32_1 = None
        convert_element_type_49: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_44: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [4096, 2048])
        permute_22: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_6: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_44, permute_22);  view_44 = permute_22 = None
        view_45: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [32, 128, 2048]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_50: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [32, 128, 16, 128]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_25: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_56: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_25, torch.float32);  permute_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_10: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_56, [32, 16, 128, 128]);  convert_element_type_56 = None
        clone_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_53: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [512, 128, 128]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_46: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [4096, 2048])
        permute_23: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_7: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_46, permute_23);  view_46 = permute_23 = None
        view_47: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [32, 128, 2048]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_51: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_47, [32, 128, 16, 128]);  view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_26: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_57: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_26, torch.float32);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_28: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_57, [0, 1, 3, 2]);  convert_element_type_57 = None
        expand_11: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_28, [32, 16, 128, 128]);  permute_28 = None
        clone_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_54: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [512, 128, 128]);  clone_16 = None
        bmm_4: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_53, view_54);  view_53 = view_54 = None
        view_55: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [32, 16, 128, 128]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_4: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_9, view_55, full_default_4);  slice_9 = view_55 = full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_24: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_3, where);  where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_2: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_24, [-1], True)
        sub_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_24, amax_2);  add_24 = amax_2 = None
        exp_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_58: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_12: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_58, [32, 16, 128, 128]);  convert_element_type_58 = None
        view_56: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_12, [512, 128, 128]);  expand_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_48: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [4096, 2048]);  convert_element_type_49 = None
        permute_24: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_8: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_48, permute_24);  view_48 = permute_24 = None
        view_49: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [32, 128, 2048]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_52: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_49, [32, 128, 16, 128]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_27: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_13: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_27, [32, 16, 128, 128]);  permute_27 = None
        clone_18: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_57: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [512, 128, 128]);  clone_18 = None
        bmm_5: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_56, view_57);  view_56 = view_57 = None
        view_58: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [32, 16, 128, 128]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_29: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_19: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_59: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [32, 128, 2048]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_60: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [4096, 2048]);  view_59 = None
        permute_30: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_6: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg38_1, view_60, permute_30);  arg38_1 = view_60 = permute_30 = None
        view_61: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [32, 128, 2048]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_25: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_61, add_21);  view_61 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_64: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_64, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_10: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_64, getitem_11);  convert_element_type_64 = getitem_11 = None
        add_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_18: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_5);  sub_10 = rsqrt_5 = None
        mul_19: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, arg39_1);  mul_18 = arg39_1 = None
        add_27: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, arg40_1);  mul_19 = arg40_1 = None
        convert_element_type_65: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_62: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_65, [4096, 2048]);  convert_element_type_65 = None
        permute_31: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_7: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_62, permute_31);  arg42_1 = view_62 = permute_31 = None
        view_63: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 128, 8192]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        pow_3: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_63, 3.0)
        mul_21: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_28: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_63, mul_21);  view_63 = mul_21 = None
        mul_22: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
        tanh_2: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_29: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, add_29);  mul_20 = add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_64: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_23, [4096, 8192]);  mul_23 = None
        permute_32: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_8: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_64, permute_32);  arg44_1 = view_64 = permute_32 = None
        view_65: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [32, 128, 2048]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_30: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_25, view_65);  add_25 = view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_72: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant4: "f32[][]cuda:0" = self._tensor_constant4;  _tensor_constant4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_10: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg50_1, 2, 0, 128);  arg50_1 = None
        slice_11: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 128);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_11: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_13);  convert_element_type_72 = getitem_13 = None
        add_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_24: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_6);  sub_11 = rsqrt_6 = None
        mul_25: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, arg45_1);  mul_24 = arg45_1 = None
        add_32: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, arg46_1);  mul_25 = arg46_1 = None
        convert_element_type_73: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_66: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [4096, 2048])
        permute_33: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_9: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_66, permute_33);  view_66 = permute_33 = None
        view_67: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [32, 128, 2048]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_72: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [32, 128, 16, 128]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_36: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_80: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_36, torch.float32);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_14: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_80, [32, 16, 128, 128]);  convert_element_type_80 = None
        clone_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_75: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [512, 128, 128]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_68: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [4096, 2048])
        permute_34: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        mm_10: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_68, permute_34);  view_68 = permute_34 = None
        view_69: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [32, 128, 2048]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_73: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_69, [32, 128, 16, 128]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_37: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_81: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_37, torch.float32);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_39: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_81, [0, 1, 3, 2]);  convert_element_type_81 = None
        expand_15: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_39, [32, 16, 128, 128]);  permute_39 = None
        clone_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_76: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [512, 128, 128]);  clone_23 = None
        bmm_6: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
        view_77: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [32, 16, 128, 128]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_5: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_4: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_11, view_77, full_default_5);  slice_11 = view_77 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_33: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_4, where);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_3: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_33, [-1], True)
        sub_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_33, amax_3);  add_33 = amax_3 = None
        exp_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_4: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_82: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_16: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_82, [32, 16, 128, 128]);  convert_element_type_82 = None
        view_78: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_16, [512, 128, 128]);  expand_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_70: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [4096, 2048]);  convert_element_type_73 = None
        permute_35: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_11: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_70, permute_35);  view_70 = permute_35 = None
        view_71: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [32, 128, 2048]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_74: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_71, [32, 128, 16, 128]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_38: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_17: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_38, [32, 16, 128, 128]);  permute_38 = None
        clone_25: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_79: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [512, 128, 128]);  clone_25 = None
        bmm_7: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
        view_80: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [32, 16, 128, 128]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_40: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_26: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_81: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [32, 128, 2048]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_82: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [4096, 2048]);  view_81 = None
        permute_41: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_9: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg52_1, view_82, permute_41);  arg52_1 = view_82 = permute_41 = None
        view_83: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [32, 128, 2048]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_34: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_83, add_30);  view_83 = add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_88: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_88, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_13: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_88, getitem_15);  convert_element_type_88 = getitem_15 = None
        add_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_26: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_7);  sub_13 = rsqrt_7 = None
        mul_27: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, arg53_1);  mul_26 = arg53_1 = None
        add_36: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, arg54_1);  mul_27 = arg54_1 = None
        convert_element_type_89: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_84: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_89, [4096, 2048]);  convert_element_type_89 = None
        permute_42: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_10: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg56_1, view_84, permute_42);  arg56_1 = view_84 = permute_42 = None
        view_85: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 128, 8192]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        pow_4: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_85, 3.0)
        mul_29: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_37: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_85, mul_29);  view_85 = mul_29 = None
        mul_30: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_37, 0.7978845608028654);  add_37 = None
        tanh_3: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_38: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, add_38);  mul_28 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_86: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_31, [4096, 8192]);  mul_31 = None
        permute_43: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_11: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg58_1, view_86, permute_43);  arg58_1 = view_86 = permute_43 = None
        view_87: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [32, 128, 2048]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_39: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_34, view_87);  add_34 = view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_96: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant5: "f32[][]cuda:0" = self._tensor_constant5;  _tensor_constant5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_12: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg64_1, 2, 0, 128);  arg64_1 = None
        slice_13: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 128);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_14: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_17);  convert_element_type_96 = getitem_17 = None
        add_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_32: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_8);  sub_14 = rsqrt_8 = None
        mul_33: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, arg59_1);  mul_32 = arg59_1 = None
        add_41: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, arg60_1);  mul_33 = arg60_1 = None
        convert_element_type_97: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_88: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [4096, 2048])
        permute_44: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_12: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_88, permute_44);  view_88 = permute_44 = None
        view_89: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [32, 128, 2048]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_94: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [32, 128, 16, 128]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_47: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_104: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_47, torch.float32);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_18: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_104, [32, 16, 128, 128]);  convert_element_type_104 = None
        clone_29: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_97: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [512, 128, 128]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_90: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [4096, 2048])
        permute_45: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        mm_13: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_90, permute_45);  view_90 = permute_45 = None
        view_91: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [32, 128, 2048]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_95: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_91, [32, 128, 16, 128]);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_48: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_105: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_48, torch.float32);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_50: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_105, [0, 1, 3, 2]);  convert_element_type_105 = None
        expand_19: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_50, [32, 16, 128, 128]);  permute_50 = None
        clone_30: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_98: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [512, 128, 128]);  clone_30 = None
        bmm_8: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
        view_99: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [32, 16, 128, 128]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_6: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_5: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_13, view_99, full_default_6);  slice_13 = view_99 = full_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_42: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_5, where);  where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_4: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_42, [-1], True)
        sub_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_42, amax_4);  add_42 = amax_4 = None
        exp_4: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_5: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_106: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_20: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_106, [32, 16, 128, 128]);  convert_element_type_106 = None
        view_100: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_20, [512, 128, 128]);  expand_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_92: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [4096, 2048]);  convert_element_type_97 = None
        permute_46: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_14: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_92, permute_46);  view_92 = permute_46 = None
        view_93: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [32, 128, 2048]);  mm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_96: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_93, [32, 128, 16, 128]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_49: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_21: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_49, [32, 16, 128, 128]);  permute_49 = None
        clone_32: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_101: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [512, 128, 128]);  clone_32 = None
        bmm_9: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
        view_102: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [32, 16, 128, 128]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_51: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_33: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_103: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [32, 128, 2048]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_104: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [4096, 2048]);  view_103 = None
        permute_52: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_12: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg66_1, view_104, permute_52);  arg66_1 = view_104 = permute_52 = None
        view_105: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [32, 128, 2048]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_43: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_105, add_39);  view_105 = add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_112: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_112, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_16: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_112, getitem_19);  convert_element_type_112 = getitem_19 = None
        add_44: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_34: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_9);  sub_16 = rsqrt_9 = None
        mul_35: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, arg67_1);  mul_34 = arg67_1 = None
        add_45: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_35, arg68_1);  mul_35 = arg68_1 = None
        convert_element_type_113: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_106: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_113, [4096, 2048]);  convert_element_type_113 = None
        permute_53: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_13: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg70_1, view_106, permute_53);  arg70_1 = view_106 = permute_53 = None
        view_107: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 128, 8192]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        pow_5: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
        mul_37: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_46: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_107, mul_37);  view_107 = mul_37 = None
        mul_38: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_4: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_47: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, add_47);  mul_36 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_108: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_39, [4096, 8192]);  mul_39 = None
        permute_54: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_14: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_108, permute_54);  arg72_1 = view_108 = permute_54 = None
        view_109: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [32, 128, 2048]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_48: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_43, view_109);  add_43 = view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_120: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant6: "f32[][]cuda:0" = self._tensor_constant6;  _tensor_constant6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_14: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg78_1, 2, 0, 128);  arg78_1 = None
        slice_15: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 128);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_17: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_21);  convert_element_type_120 = getitem_21 = None
        add_49: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_40: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_10);  sub_17 = rsqrt_10 = None
        mul_41: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, arg73_1);  mul_40 = arg73_1 = None
        add_50: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, arg74_1);  mul_41 = arg74_1 = None
        convert_element_type_121: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_110: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [4096, 2048])
        permute_55: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        mm_15: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_110, permute_55);  view_110 = permute_55 = None
        view_111: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [32, 128, 2048]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_116: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [32, 128, 16, 128]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_58: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_128: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_58, torch.float32);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_22: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_128, [32, 16, 128, 128]);  convert_element_type_128 = None
        clone_36: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_119: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [512, 128, 128]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_112: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [4096, 2048])
        permute_56: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_16: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_112, permute_56);  view_112 = permute_56 = None
        view_113: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [32, 128, 2048]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_117: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [32, 128, 16, 128]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_59: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_129: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_59, torch.float32);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_61: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_129, [0, 1, 3, 2]);  convert_element_type_129 = None
        expand_23: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_61, [32, 16, 128, 128]);  permute_61 = None
        clone_37: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_120: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [512, 128, 128]);  clone_37 = None
        bmm_10: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
        view_121: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [32, 16, 128, 128]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_7: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_15, view_121, full_default_7);  slice_15 = view_121 = full_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_51: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_6, where);  where_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_5: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_51, [-1], True)
        sub_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_51, amax_5);  add_51 = amax_5 = None
        exp_5: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_6: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_130: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_24: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_130, [32, 16, 128, 128]);  convert_element_type_130 = None
        view_122: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_24, [512, 128, 128]);  expand_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_114: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [4096, 2048]);  convert_element_type_121 = None
        permute_57: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_17: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_114, permute_57);  view_114 = permute_57 = None
        view_115: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [32, 128, 2048]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_118: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_115, [32, 128, 16, 128]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_60: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_25: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_60, [32, 16, 128, 128]);  permute_60 = None
        clone_39: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_123: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [512, 128, 128]);  clone_39 = None
        bmm_11: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
        view_124: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [32, 16, 128, 128]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_62: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_40: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_125: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [32, 128, 2048]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_126: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [4096, 2048]);  view_125 = None
        permute_63: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_15: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_126, permute_63);  arg80_1 = view_126 = permute_63 = None
        view_127: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [32, 128, 2048]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_52: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_127, add_48);  view_127 = add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_136: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_136, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_19: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_136, getitem_23);  convert_element_type_136 = getitem_23 = None
        add_53: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_42: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_11);  sub_19 = rsqrt_11 = None
        mul_43: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, arg81_1);  mul_42 = arg81_1 = None
        add_54: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, arg82_1);  mul_43 = arg82_1 = None
        convert_element_type_137: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_128: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_137, [4096, 2048]);  convert_element_type_137 = None
        permute_64: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_16: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg84_1, view_128, permute_64);  arg84_1 = view_128 = permute_64 = None
        view_129: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 128, 8192]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        pow_6: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_129, 3.0)
        mul_45: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_55: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_129, mul_45);  view_129 = mul_45 = None
        mul_46: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_5: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_56: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, add_56);  mul_44 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_130: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_47, [4096, 8192]);  mul_47 = None
        permute_65: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_17: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg86_1, view_130, permute_65);  arg86_1 = view_130 = permute_65 = None
        view_131: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [32, 128, 2048]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_57: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_52, view_131);  add_52 = view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_144: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant7: "f32[][]cuda:0" = self._tensor_constant7;  _tensor_constant7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_16: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg92_1, 2, 0, 128);  arg92_1 = None
        slice_17: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_16, 3, 0, 128);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_20: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_25);  convert_element_type_144 = getitem_25 = None
        add_58: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_48: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_12);  sub_20 = rsqrt_12 = None
        mul_49: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, arg87_1);  mul_48 = arg87_1 = None
        add_59: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, arg88_1);  mul_49 = arg88_1 = None
        convert_element_type_145: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_132: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [4096, 2048])
        permute_66: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_18: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_132, permute_66);  view_132 = permute_66 = None
        view_133: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [32, 128, 2048]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_138: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [32, 128, 16, 128]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_69: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_152: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_69, torch.float32);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_26: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_152, [32, 16, 128, 128]);  convert_element_type_152 = None
        clone_43: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_141: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [512, 128, 128]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_134: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [4096, 2048])
        permute_67: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_19: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_134, permute_67);  view_134 = permute_67 = None
        view_135: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [32, 128, 2048]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_139: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_135, [32, 128, 16, 128]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_70: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_153: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_70, torch.float32);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_72: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_153, [0, 1, 3, 2]);  convert_element_type_153 = None
        expand_27: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_72, [32, 16, 128, 128]);  permute_72 = None
        clone_44: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_142: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [512, 128, 128]);  clone_44 = None
        bmm_12: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_141, view_142);  view_141 = view_142 = None
        view_143: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [32, 16, 128, 128]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_8: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_7: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_17, view_143, full_default_8);  slice_17 = view_143 = full_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_60: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_7, where);  where_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_6: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_60, [-1], True)
        sub_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_60, amax_6);  add_60 = amax_6 = None
        exp_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_7: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_154: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_28: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_154, [32, 16, 128, 128]);  convert_element_type_154 = None
        view_144: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_28, [512, 128, 128]);  expand_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_136: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [4096, 2048]);  convert_element_type_145 = None
        permute_68: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_20: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_136, permute_68);  view_136 = permute_68 = None
        view_137: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [32, 128, 2048]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_140: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [32, 128, 16, 128]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_71: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_29: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_71, [32, 16, 128, 128]);  permute_71 = None
        clone_46: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_145: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [512, 128, 128]);  clone_46 = None
        bmm_13: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
        view_146: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [32, 16, 128, 128]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_73: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_47: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_147: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [32, 128, 2048]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_148: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [4096, 2048]);  view_147 = None
        permute_74: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_18: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg94_1, view_148, permute_74);  arg94_1 = view_148 = permute_74 = None
        view_149: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [32, 128, 2048]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_61: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_149, add_57);  view_149 = add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_160: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_160, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_22: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_160, getitem_27);  convert_element_type_160 = getitem_27 = None
        add_62: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_50: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_13);  sub_22 = rsqrt_13 = None
        mul_51: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, arg95_1);  mul_50 = arg95_1 = None
        add_63: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, arg96_1);  mul_51 = arg96_1 = None
        convert_element_type_161: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_150: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_161, [4096, 2048]);  convert_element_type_161 = None
        permute_75: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_19: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, view_150, permute_75);  arg98_1 = view_150 = permute_75 = None
        view_151: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 128, 8192]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        pow_7: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_151, 3.0)
        mul_53: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_64: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_151, mul_53);  view_151 = mul_53 = None
        mul_54: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        tanh_6: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_65: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, add_65);  mul_52 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_152: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_55, [4096, 8192]);  mul_55 = None
        permute_76: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_20: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg100_1, view_152, permute_76);  arg100_1 = view_152 = permute_76 = None
        view_153: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [32, 128, 2048]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_66: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_61, view_153);  add_61 = view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_168: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_168, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant8: "f32[][]cuda:0" = self._tensor_constant8;  _tensor_constant8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_18: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg106_1, 2, 0, 128);  arg106_1 = None
        slice_19: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_18, 3, 0, 128);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_23: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_168, getitem_29);  convert_element_type_168 = getitem_29 = None
        add_67: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_56: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_14);  sub_23 = rsqrt_14 = None
        mul_57: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, arg101_1);  mul_56 = arg101_1 = None
        add_68: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, arg102_1);  mul_57 = arg102_1 = None
        convert_element_type_169: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_154: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_169, [4096, 2048])
        permute_77: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_21: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_154, permute_77);  view_154 = permute_77 = None
        view_155: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [32, 128, 2048]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_160: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [32, 128, 16, 128]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_80: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_176: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_80, torch.float32);  permute_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_30: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_176, [32, 16, 128, 128]);  convert_element_type_176 = None
        clone_50: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_163: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [512, 128, 128]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_156: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_169, [4096, 2048])
        permute_78: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_22: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_156, permute_78);  view_156 = permute_78 = None
        view_157: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [32, 128, 2048]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_161: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_157, [32, 128, 16, 128]);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_81: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_177: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_81, torch.float32);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_83: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_177, [0, 1, 3, 2]);  convert_element_type_177 = None
        expand_31: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_83, [32, 16, 128, 128]);  permute_83 = None
        clone_51: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_164: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [512, 128, 128]);  clone_51 = None
        bmm_14: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_163, view_164);  view_163 = view_164 = None
        view_165: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [32, 16, 128, 128]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_9: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_19, view_165, full_default_9);  slice_19 = view_165 = full_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_69: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_8, where);  where_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_7: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_69, [-1], True)
        sub_24: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_69, amax_7);  add_69 = amax_7 = None
        exp_7: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_8: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_178: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_32: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_178, [32, 16, 128, 128]);  convert_element_type_178 = None
        view_166: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_32, [512, 128, 128]);  expand_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_158: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_169, [4096, 2048]);  convert_element_type_169 = None
        permute_79: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_23: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_158, permute_79);  view_158 = permute_79 = None
        view_159: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [32, 128, 2048]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_162: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_159, [32, 128, 16, 128]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_82: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_33: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_82, [32, 16, 128, 128]);  permute_82 = None
        clone_53: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_167: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [512, 128, 128]);  clone_53 = None
        bmm_15: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
        view_168: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [32, 16, 128, 128]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_84: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_54: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_169: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [32, 128, 2048]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_170: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [4096, 2048]);  view_169 = None
        permute_85: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_21: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg108_1, view_170, permute_85);  arg108_1 = view_170 = permute_85 = None
        view_171: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [32, 128, 2048]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_70: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_171, add_66);  view_171 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_184: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_184, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_25: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_184, getitem_31);  convert_element_type_184 = getitem_31 = None
        add_71: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_58: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_15);  sub_25 = rsqrt_15 = None
        mul_59: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, arg109_1);  mul_58 = arg109_1 = None
        add_72: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_59, arg110_1);  mul_59 = arg110_1 = None
        convert_element_type_185: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_172: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_185, [4096, 2048]);  convert_element_type_185 = None
        permute_86: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_22: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg112_1, view_172, permute_86);  arg112_1 = view_172 = permute_86 = None
        view_173: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 128, 8192]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        pow_8: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_173, 3.0)
        mul_61: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_73: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_173, mul_61);  view_173 = mul_61 = None
        mul_62: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_7: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_74: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, add_74);  mul_60 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_174: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_63, [4096, 8192]);  mul_63 = None
        permute_87: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_23: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg114_1, view_174, permute_87);  arg114_1 = view_174 = permute_87 = None
        view_175: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [32, 128, 2048]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_75: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_70, view_175);  add_70 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_192: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_192, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant9: "f32[][]cuda:0" = self._tensor_constant9;  _tensor_constant9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_20: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg120_1, 2, 0, 128);  arg120_1 = None
        slice_21: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 128);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_26: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_33);  convert_element_type_192 = getitem_33 = None
        add_76: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_64: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_16);  sub_26 = rsqrt_16 = None
        mul_65: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, arg115_1);  mul_64 = arg115_1 = None
        add_77: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, arg116_1);  mul_65 = arg116_1 = None
        convert_element_type_193: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_176: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [4096, 2048])
        permute_88: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_24: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_88);  view_176 = permute_88 = None
        view_177: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [32, 128, 2048]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_182: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [32, 128, 16, 128]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_91: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_200: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_91, torch.float32);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_34: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_200, [32, 16, 128, 128]);  convert_element_type_200 = None
        clone_57: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_185: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [512, 128, 128]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_178: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [4096, 2048])
        permute_89: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_25: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_178, permute_89);  view_178 = permute_89 = None
        view_179: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [32, 128, 2048]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_183: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_179, [32, 128, 16, 128]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_92: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_201: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_92, torch.float32);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_94: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_201, [0, 1, 3, 2]);  convert_element_type_201 = None
        expand_35: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_94, [32, 16, 128, 128]);  permute_94 = None
        clone_58: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_186: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [512, 128, 128]);  clone_58 = None
        bmm_16: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_185, view_186);  view_185 = view_186 = None
        view_187: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [32, 16, 128, 128]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_10: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_21, view_187, full_default_10);  slice_21 = view_187 = full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_78: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_9, where);  where_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_8: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_27: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_78, amax_8);  add_78 = amax_8 = None
        exp_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_9: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_202: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_36: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_202, [32, 16, 128, 128]);  convert_element_type_202 = None
        view_188: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_36, [512, 128, 128]);  expand_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_180: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [4096, 2048]);  convert_element_type_193 = None
        permute_90: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_26: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_180, permute_90);  view_180 = permute_90 = None
        view_181: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [32, 128, 2048]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_184: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_181, [32, 128, 16, 128]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_93: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_37: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_93, [32, 16, 128, 128]);  permute_93 = None
        clone_60: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_189: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [512, 128, 128]);  clone_60 = None
        bmm_17: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_188, view_189);  view_188 = view_189 = None
        view_190: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [32, 16, 128, 128]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_95: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_61: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_191: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [32, 128, 2048]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_192: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [4096, 2048]);  view_191 = None
        permute_96: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_24: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg122_1, view_192, permute_96);  arg122_1 = view_192 = permute_96 = None
        view_193: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [32, 128, 2048]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_79: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, add_75);  view_193 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_208: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_208, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_28: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_208, getitem_35);  convert_element_type_208 = getitem_35 = None
        add_80: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_66: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_17);  sub_28 = rsqrt_17 = None
        mul_67: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg123_1);  mul_66 = arg123_1 = None
        add_81: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg124_1);  mul_67 = arg124_1 = None
        convert_element_type_209: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_194: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_209, [4096, 2048]);  convert_element_type_209 = None
        permute_97: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_25: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg126_1, view_194, permute_97);  arg126_1 = view_194 = permute_97 = None
        view_195: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 128, 8192]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        pow_9: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_195, 3.0)
        mul_69: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_82: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_195, mul_69);  view_195 = mul_69 = None
        mul_70: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
        tanh_8: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_83: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_83);  mul_68 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_196: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_71, [4096, 8192]);  mul_71 = None
        permute_98: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_26: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg128_1, view_196, permute_98);  arg128_1 = view_196 = permute_98 = None
        view_197: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [32, 128, 2048]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_84: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_79, view_197);  add_79 = view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_216: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_216, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant10: "f32[][]cuda:0" = self._tensor_constant10;  _tensor_constant10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_22: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg134_1, 2, 0, 128);  arg134_1 = None
        slice_23: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 128);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_29: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_216, getitem_37);  convert_element_type_216 = getitem_37 = None
        add_85: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_72: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_18);  sub_29 = rsqrt_18 = None
        mul_73: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, arg129_1);  mul_72 = arg129_1 = None
        add_86: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, arg130_1);  mul_73 = arg130_1 = None
        convert_element_type_217: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_198: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_217, [4096, 2048])
        permute_99: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_27: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_99);  view_198 = permute_99 = None
        view_199: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [32, 128, 2048]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_204: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [32, 128, 16, 128]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_102: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_224: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_102, torch.float32);  permute_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_38: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_224, [32, 16, 128, 128]);  convert_element_type_224 = None
        clone_64: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_207: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [512, 128, 128]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_200: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_217, [4096, 2048])
        permute_100: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_28: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_200, permute_100);  view_200 = permute_100 = None
        view_201: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [32, 128, 2048]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_205: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_201, [32, 128, 16, 128]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_103: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_225: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_103, torch.float32);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_105: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_225, [0, 1, 3, 2]);  convert_element_type_225 = None
        expand_39: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_105, [32, 16, 128, 128]);  permute_105 = None
        clone_65: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_208: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [512, 128, 128]);  clone_65 = None
        bmm_18: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_207, view_208);  view_207 = view_208 = None
        view_209: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [32, 16, 128, 128]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_11: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_10: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_23, view_209, full_default_11);  slice_23 = view_209 = full_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_87: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_10, where);  where_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_9: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_87, [-1], True)
        sub_30: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_87, amax_9);  add_87 = amax_9 = None
        exp_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_10: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_226: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_40: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_226, [32, 16, 128, 128]);  convert_element_type_226 = None
        view_210: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_40, [512, 128, 128]);  expand_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_202: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_217, [4096, 2048]);  convert_element_type_217 = None
        permute_101: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_29: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_202, permute_101);  view_202 = permute_101 = None
        view_203: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [32, 128, 2048]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_206: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_203, [32, 128, 16, 128]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_104: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_41: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_104, [32, 16, 128, 128]);  permute_104 = None
        clone_67: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_211: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [512, 128, 128]);  clone_67 = None
        bmm_19: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
        view_212: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [32, 16, 128, 128]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_106: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_68: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_213: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [32, 128, 2048]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_214: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [4096, 2048]);  view_213 = None
        permute_107: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_27: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg136_1, view_214, permute_107);  arg136_1 = view_214 = permute_107 = None
        view_215: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [32, 128, 2048]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_88: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_215, add_84);  view_215 = add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_232: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_232, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_31: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_232, getitem_39);  convert_element_type_232 = getitem_39 = None
        add_89: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_74: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_19);  sub_31 = rsqrt_19 = None
        mul_75: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, arg137_1);  mul_74 = arg137_1 = None
        add_90: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_75, arg138_1);  mul_75 = arg138_1 = None
        convert_element_type_233: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_216: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_233, [4096, 2048]);  convert_element_type_233 = None
        permute_108: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_28: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_216, permute_108);  arg140_1 = view_216 = permute_108 = None
        view_217: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 128, 8192]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        pow_10: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_217, 3.0)
        mul_77: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_91: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_217, mul_77);  view_217 = mul_77 = None
        mul_78: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        tanh_9: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_92: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, add_92);  mul_76 = add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_218: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_79, [4096, 8192]);  mul_79 = None
        permute_109: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_29: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg142_1, view_218, permute_109);  arg142_1 = view_218 = permute_109 = None
        view_219: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [32, 128, 2048]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_93: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_88, view_219);  add_88 = view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_240: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_240, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant11: "f32[][]cuda:0" = self._tensor_constant11;  _tensor_constant11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_24: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg148_1, 2, 0, 128);  arg148_1 = None
        slice_25: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_24, 3, 0, 128);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_32: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, getitem_41);  convert_element_type_240 = getitem_41 = None
        add_94: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_80: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_20);  sub_32 = rsqrt_20 = None
        mul_81: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, arg143_1);  mul_80 = arg143_1 = None
        add_95: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, arg144_1);  mul_81 = arg144_1 = None
        convert_element_type_241: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_220: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [4096, 2048])
        permute_110: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_30: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_220, permute_110);  view_220 = permute_110 = None
        view_221: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [32, 128, 2048]);  mm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_226: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [32, 128, 16, 128]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_113: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_248: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_113, torch.float32);  permute_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_42: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_248, [32, 16, 128, 128]);  convert_element_type_248 = None
        clone_71: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_229: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [512, 128, 128]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_222: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [4096, 2048])
        permute_111: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        mm_31: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_222, permute_111);  view_222 = permute_111 = None
        view_223: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [32, 128, 2048]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_227: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_223, [32, 128, 16, 128]);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_114: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_249: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_114, torch.float32);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_116: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_249, [0, 1, 3, 2]);  convert_element_type_249 = None
        expand_43: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_116, [32, 16, 128, 128]);  permute_116 = None
        clone_72: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_230: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [512, 128, 128]);  clone_72 = None
        bmm_20: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_229, view_230);  view_229 = view_230 = None
        view_231: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [32, 16, 128, 128]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_12: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_25, view_231, full_default_12);  slice_25 = view_231 = full_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_96: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_11, where);  where_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_10: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_96, [-1], True)
        sub_33: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_96, amax_10);  add_96 = amax_10 = None
        exp_10: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_11: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_250: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_44: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_250, [32, 16, 128, 128]);  convert_element_type_250 = None
        view_232: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_44, [512, 128, 128]);  expand_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_224: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [4096, 2048]);  convert_element_type_241 = None
        permute_112: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_32: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_224, permute_112);  view_224 = permute_112 = None
        view_225: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [32, 128, 2048]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_228: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [32, 128, 16, 128]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_115: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_45: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_115, [32, 16, 128, 128]);  permute_115 = None
        clone_74: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_233: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [512, 128, 128]);  clone_74 = None
        bmm_21: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = view_233 = None
        view_234: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [32, 16, 128, 128]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_117: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_75: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_235: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [32, 128, 2048]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_236: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [4096, 2048]);  view_235 = None
        permute_118: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_30: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg150_1, view_236, permute_118);  arg150_1 = view_236 = permute_118 = None
        view_237: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [32, 128, 2048]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_97: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_237, add_93);  view_237 = add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_256: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_256, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_34: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_256, getitem_43);  convert_element_type_256 = getitem_43 = None
        add_98: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_82: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_21);  sub_34 = rsqrt_21 = None
        mul_83: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, arg151_1);  mul_82 = arg151_1 = None
        add_99: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, arg152_1);  mul_83 = arg152_1 = None
        convert_element_type_257: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_238: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_257, [4096, 2048]);  convert_element_type_257 = None
        permute_119: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_31: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg154_1, view_238, permute_119);  arg154_1 = view_238 = permute_119 = None
        view_239: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 128, 8192]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        pow_11: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_239, 3.0)
        mul_85: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_100: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_239, mul_85);  view_239 = mul_85 = None
        mul_86: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_101: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, add_101);  mul_84 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_240: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_87, [4096, 8192]);  mul_87 = None
        permute_120: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_32: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg156_1, view_240, permute_120);  arg156_1 = view_240 = permute_120 = None
        view_241: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [32, 128, 2048]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_102: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, view_241);  add_97 = view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_264: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_264, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant12: "f32[][]cuda:0" = self._tensor_constant12;  _tensor_constant12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_26: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg162_1, 2, 0, 128);  arg162_1 = None
        slice_27: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_26, 3, 0, 128);  slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_35: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_264, getitem_45);  convert_element_type_264 = getitem_45 = None
        add_103: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_88: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_22);  sub_35 = rsqrt_22 = None
        mul_89: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, arg157_1);  mul_88 = arg157_1 = None
        add_104: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_89, arg158_1);  mul_89 = arg158_1 = None
        convert_element_type_265: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_242: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [4096, 2048])
        permute_121: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_33: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_121);  view_242 = permute_121 = None
        view_243: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [32, 128, 2048]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_248: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_243, [32, 128, 16, 128]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_124: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_272: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_124, torch.float32);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_46: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_272, [32, 16, 128, 128]);  convert_element_type_272 = None
        clone_78: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_251: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [512, 128, 128]);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_244: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [4096, 2048])
        permute_122: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_34: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_244, permute_122);  view_244 = permute_122 = None
        view_245: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [32, 128, 2048]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_249: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [32, 128, 16, 128]);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_125: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_273: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_125, torch.float32);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_127: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_273, [0, 1, 3, 2]);  convert_element_type_273 = None
        expand_47: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_127, [32, 16, 128, 128]);  permute_127 = None
        clone_79: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_252: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [512, 128, 128]);  clone_79 = None
        bmm_22: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
        view_253: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [32, 16, 128, 128]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_13: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_27, view_253, full_default_13);  slice_27 = view_253 = full_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_105: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_12, where);  where_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_11: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_105, [-1], True)
        sub_36: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_105, amax_11);  add_105 = amax_11 = None
        exp_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_12: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_274: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_48: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_274, [32, 16, 128, 128]);  convert_element_type_274 = None
        view_254: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_48, [512, 128, 128]);  expand_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_246: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [4096, 2048]);  convert_element_type_265 = None
        permute_123: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        mm_35: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_246, permute_123);  view_246 = permute_123 = None
        view_247: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [32, 128, 2048]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_250: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_247, [32, 128, 16, 128]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_126: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_49: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_126, [32, 16, 128, 128]);  permute_126 = None
        clone_81: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_255: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [512, 128, 128]);  clone_81 = None
        bmm_23: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
        view_256: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [32, 16, 128, 128]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_128: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_82: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_257: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [32, 128, 2048]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_258: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [4096, 2048]);  view_257 = None
        permute_129: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_33: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg164_1, view_258, permute_129);  arg164_1 = view_258 = permute_129 = None
        view_259: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [32, 128, 2048]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_106: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_259, add_102);  view_259 = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_280: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_280, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_37: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_280, getitem_47);  convert_element_type_280 = getitem_47 = None
        add_107: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_90: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_23);  sub_37 = rsqrt_23 = None
        mul_91: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, arg165_1);  mul_90 = arg165_1 = None
        add_108: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, arg166_1);  mul_91 = arg166_1 = None
        convert_element_type_281: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16);  add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_260: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_281, [4096, 2048]);  convert_element_type_281 = None
        permute_130: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_34: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg168_1, view_260, permute_130);  arg168_1 = view_260 = permute_130 = None
        view_261: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 128, 8192]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        pow_12: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_261, 3.0)
        mul_93: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_109: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_261, mul_93);  view_261 = mul_93 = None
        mul_94: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_109, 0.7978845608028654);  add_109 = None
        tanh_11: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_110: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, add_110);  mul_92 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_262: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_95, [4096, 8192]);  mul_95 = None
        permute_131: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_35: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg170_1, view_262, permute_131);  arg170_1 = view_262 = permute_131 = None
        view_263: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [32, 128, 2048]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_111: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_106, view_263);  add_106 = view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_288: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_288, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant13: "f32[][]cuda:0" = self._tensor_constant13;  _tensor_constant13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_28: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg176_1, 2, 0, 128);  arg176_1 = None
        slice_29: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_28, 3, 0, 128);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_38: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_288, getitem_49);  convert_element_type_288 = getitem_49 = None
        add_112: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_96: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_24);  sub_38 = rsqrt_24 = None
        mul_97: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, arg171_1);  mul_96 = arg171_1 = None
        add_113: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, arg172_1);  mul_97 = arg172_1 = None
        convert_element_type_289: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.bfloat16);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_264: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [4096, 2048])
        permute_132: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_36: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_264, permute_132);  view_264 = permute_132 = None
        view_265: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [32, 128, 2048]);  mm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_270: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_265, [32, 128, 16, 128]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_135: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_296: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_135, torch.float32);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_50: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_296, [32, 16, 128, 128]);  convert_element_type_296 = None
        clone_85: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_273: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [512, 128, 128]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_266: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [4096, 2048])
        permute_133: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        mm_37: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_266, permute_133);  view_266 = permute_133 = None
        view_267: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [32, 128, 2048]);  mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_271: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_267, [32, 128, 16, 128]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_136: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_297: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_136, torch.float32);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_138: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_297, [0, 1, 3, 2]);  convert_element_type_297 = None
        expand_51: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_138, [32, 16, 128, 128]);  permute_138 = None
        clone_86: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_274: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [512, 128, 128]);  clone_86 = None
        bmm_24: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_273, view_274);  view_273 = view_274 = None
        view_275: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [32, 16, 128, 128]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_14: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_13: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_29, view_275, full_default_14);  slice_29 = view_275 = full_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_114: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_13, where);  where_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_12: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_114, [-1], True)
        sub_39: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_114, amax_12);  add_114 = amax_12 = None
        exp_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_13: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_298: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_52: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_298, [32, 16, 128, 128]);  convert_element_type_298 = None
        view_276: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_52, [512, 128, 128]);  expand_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_268: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [4096, 2048]);  convert_element_type_289 = None
        permute_134: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_38: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_268, permute_134);  view_268 = permute_134 = None
        view_269: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [32, 128, 2048]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_272: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_269, [32, 128, 16, 128]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_137: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_53: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_137, [32, 16, 128, 128]);  permute_137 = None
        clone_88: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_277: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [512, 128, 128]);  clone_88 = None
        bmm_25: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_276, view_277);  view_276 = view_277 = None
        view_278: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [32, 16, 128, 128]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_139: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
        clone_89: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_279: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [32, 128, 2048]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_280: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_279, [4096, 2048]);  view_279 = None
        permute_140: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_36: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg178_1, view_280, permute_140);  arg178_1 = view_280 = permute_140 = None
        view_281: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [32, 128, 2048]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_115: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_281, add_111);  view_281 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_304: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_304, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        sub_40: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_304, getitem_51);  convert_element_type_304 = getitem_51 = None
        add_116: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_98: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_25);  sub_40 = rsqrt_25 = None
        mul_99: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, arg179_1);  mul_98 = arg179_1 = None
        add_117: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, arg180_1);  mul_99 = arg180_1 = None
        convert_element_type_305: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_282: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_305, [4096, 2048]);  convert_element_type_305 = None
        permute_141: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_37: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg182_1, view_282, permute_141);  arg182_1 = view_282 = permute_141 = None
        view_283: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 128, 8192]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_100: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_283, 0.5)
        pow_13: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_283, 3.0)
        mul_101: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_118: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_283, mul_101);  view_283 = mul_101 = None
        mul_102: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, 0.7978845608028654);  add_118 = None
        tanh_12: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
        add_119: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_103: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, add_119);  mul_100 = add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_284: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_103, [4096, 8192]);  mul_103 = None
        permute_142: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_38: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg184_1, view_284, permute_142);  arg184_1 = view_284 = permute_142 = None
        view_285: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [32, 128, 2048]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_120: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, view_285);  add_115 = view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_312: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_312, [2], correction = 0, keepdim = True)
        getitem_52: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant14: "f32[][]cuda:0" = self._tensor_constant14;  _tensor_constant14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_30: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg190_1, 2, 0, 128);  arg190_1 = None
        slice_31: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_30, 3, 0, 128);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_41: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_312, getitem_53);  convert_element_type_312 = getitem_53 = None
        add_121: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        mul_104: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_26);  sub_41 = rsqrt_26 = None
        mul_105: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, arg185_1);  mul_104 = arg185_1 = None
        add_122: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_105, arg186_1);  mul_105 = arg186_1 = None
        convert_element_type_313: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16);  add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_286: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_313, [4096, 2048])
        permute_143: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        mm_39: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_286, permute_143);  view_286 = permute_143 = None
        view_287: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [32, 128, 2048]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_292: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_287, [32, 128, 16, 128]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_146: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_320: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_146, torch.float32);  permute_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_54: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_320, [32, 16, 128, 128]);  convert_element_type_320 = None
        clone_92: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_295: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [512, 128, 128]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_288: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_313, [4096, 2048])
        permute_144: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        mm_40: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_288, permute_144);  view_288 = permute_144 = None
        view_289: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [32, 128, 2048]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_293: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_289, [32, 128, 16, 128]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_147: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_321: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_147, torch.float32);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_149: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_321, [0, 1, 3, 2]);  convert_element_type_321 = None
        expand_55: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_149, [32, 16, 128, 128]);  permute_149 = None
        clone_93: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_296: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [512, 128, 128]);  clone_93 = None
        bmm_26: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_295, view_296);  view_295 = view_296 = None
        view_297: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [32, 16, 128, 128]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_15: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_14: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_31, view_297, full_default_15);  slice_31 = view_297 = full_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_123: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_14, where);  where_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_13: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_123, [-1], True)
        sub_42: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_123, amax_13);  add_123 = amax_13 = None
        exp_13: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_14: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_322: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_56: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_322, [32, 16, 128, 128]);  convert_element_type_322 = None
        view_298: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_56, [512, 128, 128]);  expand_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_290: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_313, [4096, 2048]);  convert_element_type_313 = None
        permute_145: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_41: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_290, permute_145);  view_290 = permute_145 = None
        view_291: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [32, 128, 2048]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_294: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_291, [32, 128, 16, 128]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_148: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_57: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_148, [32, 16, 128, 128]);  permute_148 = None
        clone_95: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_299: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [512, 128, 128]);  clone_95 = None
        bmm_27: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_298, view_299);  view_298 = view_299 = None
        view_300: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [32, 16, 128, 128]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_150: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_96: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_301: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [32, 128, 2048]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_302: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [4096, 2048]);  view_301 = None
        permute_151: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_39: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, view_302, permute_151);  arg192_1 = view_302 = permute_151 = None
        view_303: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [32, 128, 2048]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_124: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_303, add_120);  view_303 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_328: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_328, [2], correction = 0, keepdim = True)
        getitem_54: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        sub_43: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_328, getitem_55);  convert_element_type_328 = getitem_55 = None
        add_125: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_106: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_27);  sub_43 = rsqrt_27 = None
        mul_107: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, arg193_1);  mul_106 = arg193_1 = None
        add_126: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_107, arg194_1);  mul_107 = arg194_1 = None
        convert_element_type_329: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_126, torch.bfloat16);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_304: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_329, [4096, 2048]);  convert_element_type_329 = None
        permute_152: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_40: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg196_1, view_304, permute_152);  arg196_1 = view_304 = permute_152 = None
        view_305: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 128, 8192]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_108: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        pow_14: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_305, 3.0)
        mul_109: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_127: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_305, mul_109);  view_305 = mul_109 = None
        mul_110: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_127, 0.7978845608028654);  add_127 = None
        tanh_13: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
        add_128: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_111: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, add_128);  mul_108 = add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_306: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_111, [4096, 8192]);  mul_111 = None
        permute_153: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_41: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg198_1, view_306, permute_153);  arg198_1 = view_306 = permute_153 = None
        view_307: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [32, 128, 2048]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_129: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_124, view_307);  add_124 = view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_336: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_336, [2], correction = 0, keepdim = True)
        getitem_56: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant15: "f32[][]cuda:0" = self._tensor_constant15;  _tensor_constant15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_32: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg204_1, 2, 0, 128);  arg204_1 = None
        slice_33: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_32, 3, 0, 128);  slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_44: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_336, getitem_57);  convert_element_type_336 = getitem_57 = None
        add_130: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_112: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_28);  sub_44 = rsqrt_28 = None
        mul_113: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, arg199_1);  mul_112 = arg199_1 = None
        add_131: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, arg200_1);  mul_113 = arg200_1 = None
        convert_element_type_337: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_308: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [4096, 2048])
        permute_154: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        mm_42: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_308, permute_154);  view_308 = permute_154 = None
        view_309: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [32, 128, 2048]);  mm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_314: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [32, 128, 16, 128]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_157: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_344: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_157, torch.float32);  permute_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_58: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_344, [32, 16, 128, 128]);  convert_element_type_344 = None
        clone_99: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_317: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [512, 128, 128]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_310: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [4096, 2048])
        permute_155: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        mm_43: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_310, permute_155);  view_310 = permute_155 = None
        view_311: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [32, 128, 2048]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_315: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_311, [32, 128, 16, 128]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_158: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_345: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_158, torch.float32);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_160: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_345, [0, 1, 3, 2]);  convert_element_type_345 = None
        expand_59: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_160, [32, 16, 128, 128]);  permute_160 = None
        clone_100: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_318: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [512, 128, 128]);  clone_100 = None
        bmm_28: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_317, view_318);  view_317 = view_318 = None
        view_319: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [32, 16, 128, 128]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_16: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_33, view_319, full_default_16);  slice_33 = view_319 = full_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_132: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_15, where);  where_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_14: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_132, [-1], True)
        sub_45: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_132, amax_14);  add_132 = amax_14 = None
        exp_14: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_15: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_346: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_60: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_346, [32, 16, 128, 128]);  convert_element_type_346 = None
        view_320: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_60, [512, 128, 128]);  expand_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_312: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [4096, 2048]);  convert_element_type_337 = None
        permute_156: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        mm_44: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_312, permute_156);  view_312 = permute_156 = None
        view_313: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [32, 128, 2048]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_316: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_313, [32, 128, 16, 128]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_159: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_61: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_159, [32, 16, 128, 128]);  permute_159 = None
        clone_102: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_321: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [512, 128, 128]);  clone_102 = None
        bmm_29: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_320, view_321);  view_320 = view_321 = None
        view_322: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [32, 16, 128, 128]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_161: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        clone_103: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_323: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [32, 128, 2048]);  clone_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_324: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_323, [4096, 2048]);  view_323 = None
        permute_162: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_42: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg206_1, view_324, permute_162);  arg206_1 = view_324 = permute_162 = None
        view_325: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [32, 128, 2048]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_133: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_325, add_129);  view_325 = add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_352: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_352, [2], correction = 0, keepdim = True)
        getitem_58: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        sub_46: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_352, getitem_59);  convert_element_type_352 = getitem_59 = None
        add_134: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_114: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_29);  sub_46 = rsqrt_29 = None
        mul_115: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, arg207_1);  mul_114 = arg207_1 = None
        add_135: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, arg208_1);  mul_115 = arg208_1 = None
        convert_element_type_353: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_326: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_353, [4096, 2048]);  convert_element_type_353 = None
        permute_163: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_43: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg210_1, view_326, permute_163);  arg210_1 = view_326 = permute_163 = None
        view_327: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [32, 128, 8192]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        pow_15: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_327, 3.0)
        mul_117: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_136: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_327, mul_117);  view_327 = mul_117 = None
        mul_118: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_136, 0.7978845608028654);  add_136 = None
        tanh_14: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_137: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_119: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, add_137);  mul_116 = add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_328: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_119, [4096, 8192]);  mul_119 = None
        permute_164: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_44: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg212_1, view_328, permute_164);  arg212_1 = view_328 = permute_164 = None
        view_329: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [32, 128, 2048]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_138: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, view_329);  add_133 = view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_360: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_360, [2], correction = 0, keepdim = True)
        getitem_60: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant16: "f32[][]cuda:0" = self._tensor_constant16;  _tensor_constant16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_34: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg218_1, 2, 0, 128);  arg218_1 = None
        slice_35: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_34, 3, 0, 128);  slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_47: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_360, getitem_61);  convert_element_type_360 = getitem_61 = None
        add_139: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_120: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_30);  sub_47 = rsqrt_30 = None
        mul_121: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, arg213_1);  mul_120 = arg213_1 = None
        add_140: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, arg214_1);  mul_121 = arg214_1 = None
        convert_element_type_361: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_140, torch.bfloat16);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_330: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [4096, 2048])
        permute_165: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        mm_45: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_330, permute_165);  view_330 = permute_165 = None
        view_331: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [32, 128, 2048]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_336: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_331, [32, 128, 16, 128]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_168: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_368: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_168, torch.float32);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_62: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_368, [32, 16, 128, 128]);  convert_element_type_368 = None
        clone_106: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_339: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [512, 128, 128]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_332: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [4096, 2048])
        permute_166: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        mm_46: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_332, permute_166);  view_332 = permute_166 = None
        view_333: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [32, 128, 2048]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_337: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_333, [32, 128, 16, 128]);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_169: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_369: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_169, torch.float32);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_171: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_369, [0, 1, 3, 2]);  convert_element_type_369 = None
        expand_63: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_171, [32, 16, 128, 128]);  permute_171 = None
        clone_107: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_340: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [512, 128, 128]);  clone_107 = None
        bmm_30: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_339, view_340);  view_339 = view_340 = None
        view_341: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [32, 16, 128, 128]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_17: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_35, view_341, full_default_17);  slice_35 = view_341 = full_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_141: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_16, where);  where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_15: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_141, [-1], True)
        sub_48: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_141, amax_15);  add_141 = amax_15 = None
        exp_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
        sum_16: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_370: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_64: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_370, [32, 16, 128, 128]);  convert_element_type_370 = None
        view_342: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_64, [512, 128, 128]);  expand_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_334: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [4096, 2048]);  convert_element_type_361 = None
        permute_167: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        mm_47: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_334, permute_167);  view_334 = permute_167 = None
        view_335: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [32, 128, 2048]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_338: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_335, [32, 128, 16, 128]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_170: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_65: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_170, [32, 16, 128, 128]);  permute_170 = None
        clone_109: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_343: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [512, 128, 128]);  clone_109 = None
        bmm_31: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_342, view_343);  view_342 = view_343 = None
        view_344: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [32, 16, 128, 128]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_172: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_110: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_345: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [32, 128, 2048]);  clone_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_346: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [4096, 2048]);  view_345 = None
        permute_173: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_45: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg220_1, view_346, permute_173);  arg220_1 = view_346 = permute_173 = None
        view_347: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [32, 128, 2048]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_142: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_347, add_138);  view_347 = add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_376: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_376, [2], correction = 0, keepdim = True)
        getitem_62: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        sub_49: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_376, getitem_63);  convert_element_type_376 = getitem_63 = None
        add_143: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        mul_122: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_31);  sub_49 = rsqrt_31 = None
        mul_123: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, arg221_1);  mul_122 = arg221_1 = None
        add_144: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, arg222_1);  mul_123 = arg222_1 = None
        convert_element_type_377: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_348: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_377, [4096, 2048]);  convert_element_type_377 = None
        permute_174: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_46: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg224_1, view_348, permute_174);  arg224_1 = view_348 = permute_174 = None
        view_349: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 128, 8192]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_124: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_349, 0.5)
        pow_16: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_349, 3.0)
        mul_125: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_145: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_349, mul_125);  view_349 = mul_125 = None
        mul_126: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_15: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_146: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_127: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, add_146);  mul_124 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_350: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_127, [4096, 8192]);  mul_127 = None
        permute_175: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_47: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg226_1, view_350, permute_175);  arg226_1 = view_350 = permute_175 = None
        view_351: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [32, 128, 2048]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_147: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, view_351);  add_142 = view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_384: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_384, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant17: "f32[][]cuda:0" = self._tensor_constant17;  _tensor_constant17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_36: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg232_1, 2, 0, 128);  arg232_1 = None
        slice_37: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 128);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_50: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_384, getitem_65);  convert_element_type_384 = getitem_65 = None
        add_148: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_128: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_32);  sub_50 = rsqrt_32 = None
        mul_129: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, arg227_1);  mul_128 = arg227_1 = None
        add_149: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_129, arg228_1);  mul_129 = arg228_1 = None
        convert_element_type_385: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_352: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_385, [4096, 2048])
        permute_176: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        mm_48: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_352, permute_176);  view_352 = permute_176 = None
        view_353: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [32, 128, 2048]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_358: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_353, [32, 128, 16, 128]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_179: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_392: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_179, torch.float32);  permute_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_66: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_392, [32, 16, 128, 128]);  convert_element_type_392 = None
        clone_113: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_361: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [512, 128, 128]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_354: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_385, [4096, 2048])
        permute_177: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        mm_49: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_354, permute_177);  view_354 = permute_177 = None
        view_355: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [32, 128, 2048]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_359: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_355, [32, 128, 16, 128]);  view_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_180: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_393: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_180, torch.float32);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_182: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_393, [0, 1, 3, 2]);  convert_element_type_393 = None
        expand_67: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_182, [32, 16, 128, 128]);  permute_182 = None
        clone_114: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_362: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [512, 128, 128]);  clone_114 = None
        bmm_32: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_361, view_362);  view_361 = view_362 = None
        view_363: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [32, 16, 128, 128]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_18: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_17: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_37, view_363, full_default_18);  slice_37 = view_363 = full_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_150: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_17, where);  where_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_16: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_150, [-1], True)
        sub_51: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_150, amax_16);  add_150 = amax_16 = None
        exp_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        sum_17: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_394: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_68: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_394, [32, 16, 128, 128]);  convert_element_type_394 = None
        view_364: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_68, [512, 128, 128]);  expand_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_356: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_385, [4096, 2048]);  convert_element_type_385 = None
        permute_178: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        mm_50: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_356, permute_178);  view_356 = permute_178 = None
        view_357: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [32, 128, 2048]);  mm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_360: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_357, [32, 128, 16, 128]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_181: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_69: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_181, [32, 16, 128, 128]);  permute_181 = None
        clone_116: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_365: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [512, 128, 128]);  clone_116 = None
        bmm_33: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_364, view_365);  view_364 = view_365 = None
        view_366: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [32, 16, 128, 128]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_183: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        clone_117: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_367: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [32, 128, 2048]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_368: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_367, [4096, 2048]);  view_367 = None
        permute_184: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_48: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg234_1, view_368, permute_184);  arg234_1 = view_368 = permute_184 = None
        view_369: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [32, 128, 2048]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_151: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_369, add_147);  view_369 = add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_400: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_400, [2], correction = 0, keepdim = True)
        getitem_66: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        sub_52: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_400, getitem_67);  convert_element_type_400 = getitem_67 = None
        add_152: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_130: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_33);  sub_52 = rsqrt_33 = None
        mul_131: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, arg235_1);  mul_130 = arg235_1 = None
        add_153: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, arg236_1);  mul_131 = arg236_1 = None
        convert_element_type_401: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_370: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_401, [4096, 2048]);  convert_element_type_401 = None
        permute_185: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_49: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg238_1, view_370, permute_185);  arg238_1 = view_370 = permute_185 = None
        view_371: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [32, 128, 8192]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_132: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        pow_17: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_371, 3.0)
        mul_133: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_154: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_371, mul_133);  view_371 = mul_133 = None
        mul_134: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_154, 0.7978845608028654);  add_154 = None
        tanh_16: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
        add_155: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_135: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, add_155);  mul_132 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_372: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_135, [4096, 8192]);  mul_135 = None
        permute_186: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_50: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg240_1, view_372, permute_186);  arg240_1 = view_372 = permute_186 = None
        view_373: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [32, 128, 2048]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_156: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_151, view_373);  add_151 = view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_408: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_156, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_408, [2], correction = 0, keepdim = True)
        getitem_68: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant18: "f32[][]cuda:0" = self._tensor_constant18;  _tensor_constant18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_38: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg246_1, 2, 0, 128);  arg246_1 = None
        slice_39: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 128);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_53: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_408, getitem_69);  convert_element_type_408 = getitem_69 = None
        add_157: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        mul_136: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_34);  sub_53 = rsqrt_34 = None
        mul_137: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, arg241_1);  mul_136 = arg241_1 = None
        add_158: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, arg242_1);  mul_137 = arg242_1 = None
        convert_element_type_409: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_374: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_409, [4096, 2048])
        permute_187: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        mm_51: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_374, permute_187);  view_374 = permute_187 = None
        view_375: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [32, 128, 2048]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_380: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_375, [32, 128, 16, 128]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_190: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_416: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_190, torch.float32);  permute_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_70: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_416, [32, 16, 128, 128]);  convert_element_type_416 = None
        clone_120: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_383: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [512, 128, 128]);  clone_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_376: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_409, [4096, 2048])
        permute_188: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        mm_52: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_376, permute_188);  view_376 = permute_188 = None
        view_377: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [32, 128, 2048]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_381: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_377, [32, 128, 16, 128]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_191: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_417: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_191, torch.float32);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_193: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_417, [0, 1, 3, 2]);  convert_element_type_417 = None
        expand_71: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_193, [32, 16, 128, 128]);  permute_193 = None
        clone_121: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_384: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [512, 128, 128]);  clone_121 = None
        bmm_34: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_383, view_384);  view_383 = view_384 = None
        view_385: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [32, 16, 128, 128]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_19: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_39, view_385, full_default_19);  slice_39 = view_385 = full_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_159: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_18, where);  where_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_17: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_159, [-1], True)
        sub_54: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_159, amax_17);  add_159 = amax_17 = None
        exp_17: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
        sum_18: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_418: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_72: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_418, [32, 16, 128, 128]);  convert_element_type_418 = None
        view_386: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_72, [512, 128, 128]);  expand_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_378: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_409, [4096, 2048]);  convert_element_type_409 = None
        permute_189: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        mm_53: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_378, permute_189);  view_378 = permute_189 = None
        view_379: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [32, 128, 2048]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_382: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_379, [32, 128, 16, 128]);  view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_192: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_73: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_192, [32, 16, 128, 128]);  permute_192 = None
        clone_123: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_387: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [512, 128, 128]);  clone_123 = None
        bmm_35: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_386, view_387);  view_386 = view_387 = None
        view_388: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [32, 16, 128, 128]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_194: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        clone_124: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_389: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [32, 128, 2048]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_390: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_389, [4096, 2048]);  view_389 = None
        permute_195: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_51: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg248_1, view_390, permute_195);  arg248_1 = view_390 = permute_195 = None
        view_391: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [32, 128, 2048]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_160: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_391, add_156);  view_391 = add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_424: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_160, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_424, [2], correction = 0, keepdim = True)
        getitem_70: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        sub_55: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_424, getitem_71);  convert_element_type_424 = getitem_71 = None
        add_161: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        mul_138: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_35);  sub_55 = rsqrt_35 = None
        mul_139: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, arg249_1);  mul_138 = arg249_1 = None
        add_162: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, arg250_1);  mul_139 = arg250_1 = None
        convert_element_type_425: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_162, torch.bfloat16);  add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_392: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_425, [4096, 2048]);  convert_element_type_425 = None
        permute_196: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_52: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg252_1, view_392, permute_196);  arg252_1 = view_392 = permute_196 = None
        view_393: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 128, 8192]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_393, 0.5)
        pow_18: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_393, 3.0)
        mul_141: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_163: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_393, mul_141);  view_393 = mul_141 = None
        mul_142: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_163, 0.7978845608028654);  add_163 = None
        tanh_17: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_164: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_143: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, add_164);  mul_140 = add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_394: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_143, [4096, 8192]);  mul_143 = None
        permute_197: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_53: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg254_1, view_394, permute_197);  arg254_1 = view_394 = permute_197 = None
        view_395: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [32, 128, 2048]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_165: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_160, view_395);  add_160 = view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_432: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_165, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_432, [2], correction = 0, keepdim = True)
        getitem_72: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant19: "f32[][]cuda:0" = self._tensor_constant19;  _tensor_constant19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_40: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg260_1, 2, 0, 128);  arg260_1 = None
        slice_41: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_40, 3, 0, 128);  slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_56: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_432, getitem_73);  convert_element_type_432 = getitem_73 = None
        add_166: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_144: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_36);  sub_56 = rsqrt_36 = None
        mul_145: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, arg255_1);  mul_144 = arg255_1 = None
        add_167: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_145, arg256_1);  mul_145 = arg256_1 = None
        convert_element_type_433: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.bfloat16);  add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_396: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_433, [4096, 2048])
        permute_198: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        mm_54: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_396, permute_198);  view_396 = permute_198 = None
        view_397: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [32, 128, 2048]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_402: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [32, 128, 16, 128]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_201: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_440: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_201, torch.float32);  permute_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_74: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_440, [32, 16, 128, 128]);  convert_element_type_440 = None
        clone_127: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_405: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [512, 128, 128]);  clone_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_398: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_433, [4096, 2048])
        permute_199: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        mm_55: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_398, permute_199);  view_398 = permute_199 = None
        view_399: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [32, 128, 2048]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_403: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_399, [32, 128, 16, 128]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_202: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_441: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_202, torch.float32);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_204: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_441, [0, 1, 3, 2]);  convert_element_type_441 = None
        expand_75: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_204, [32, 16, 128, 128]);  permute_204 = None
        clone_128: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_406: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [512, 128, 128]);  clone_128 = None
        bmm_36: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_405, view_406);  view_405 = view_406 = None
        view_407: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [32, 16, 128, 128]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_20: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_19: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_41, view_407, full_default_20);  slice_41 = view_407 = full_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_168: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_19, where);  where_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_18: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_168, [-1], True)
        sub_57: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_168, amax_18);  add_168 = amax_18 = None
        exp_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_19: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_442: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_76: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_442, [32, 16, 128, 128]);  convert_element_type_442 = None
        view_408: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_76, [512, 128, 128]);  expand_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_400: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_433, [4096, 2048]);  convert_element_type_433 = None
        permute_200: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        mm_56: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_400, permute_200);  view_400 = permute_200 = None
        view_401: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [32, 128, 2048]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_404: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_401, [32, 128, 16, 128]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_203: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_77: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_203, [32, 16, 128, 128]);  permute_203 = None
        clone_130: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_409: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [512, 128, 128]);  clone_130 = None
        bmm_37: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = view_409 = None
        view_410: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [32, 16, 128, 128]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_205: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_131: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_411: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [32, 128, 2048]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_412: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [4096, 2048]);  view_411 = None
        permute_206: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_54: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg262_1, view_412, permute_206);  arg262_1 = view_412 = permute_206 = None
        view_413: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [32, 128, 2048]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_169: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_413, add_165);  view_413 = add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_448: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_448, [2], correction = 0, keepdim = True)
        getitem_74: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        sub_58: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_448, getitem_75);  convert_element_type_448 = getitem_75 = None
        add_170: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_146: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_37);  sub_58 = rsqrt_37 = None
        mul_147: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, arg263_1);  mul_146 = arg263_1 = None
        add_171: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_147, arg264_1);  mul_147 = arg264_1 = None
        convert_element_type_449: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16);  add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_414: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_449, [4096, 2048]);  convert_element_type_449 = None
        permute_207: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_55: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg266_1, view_414, permute_207);  arg266_1 = view_414 = permute_207 = None
        view_415: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [32, 128, 8192]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_148: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        pow_19: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_415, 3.0)
        mul_149: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_172: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_415, mul_149);  view_415 = mul_149 = None
        mul_150: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_172, 0.7978845608028654);  add_172 = None
        tanh_18: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_173: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_151: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, add_173);  mul_148 = add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_416: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_151, [4096, 8192]);  mul_151 = None
        permute_208: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_56: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg268_1, view_416, permute_208);  arg268_1 = view_416 = permute_208 = None
        view_417: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [32, 128, 2048]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_174: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_169, view_417);  add_169 = view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_456: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_456, [2], correction = 0, keepdim = True)
        getitem_76: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant20: "f32[][]cuda:0" = self._tensor_constant20;  _tensor_constant20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_42: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg274_1, 2, 0, 128);  arg274_1 = None
        slice_43: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_42, 3, 0, 128);  slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_59: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_456, getitem_77);  convert_element_type_456 = getitem_77 = None
        add_175: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        mul_152: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_38);  sub_59 = rsqrt_38 = None
        mul_153: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, arg269_1);  mul_152 = arg269_1 = None
        add_176: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_153, arg270_1);  mul_153 = arg270_1 = None
        convert_element_type_457: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16);  add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_418: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [4096, 2048])
        permute_209: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        mm_57: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_418, permute_209);  view_418 = permute_209 = None
        view_419: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [32, 128, 2048]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_424: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_419, [32, 128, 16, 128]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_212: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_464: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_212, torch.float32);  permute_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_78: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_464, [32, 16, 128, 128]);  convert_element_type_464 = None
        clone_134: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_427: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_134, [512, 128, 128]);  clone_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_420: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [4096, 2048])
        permute_210: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        mm_58: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_420, permute_210);  view_420 = permute_210 = None
        view_421: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [32, 128, 2048]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_425: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_421, [32, 128, 16, 128]);  view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_213: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_465: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_213, torch.float32);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_215: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_465, [0, 1, 3, 2]);  convert_element_type_465 = None
        expand_79: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_215, [32, 16, 128, 128]);  permute_215 = None
        clone_135: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_428: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [512, 128, 128]);  clone_135 = None
        bmm_38: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_427, view_428);  view_427 = view_428 = None
        view_429: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [32, 16, 128, 128]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_21: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_20: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_43, view_429, full_default_21);  slice_43 = view_429 = full_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_177: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_20, where);  where_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_19: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_177, [-1], True)
        sub_60: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_177, amax_19);  add_177 = amax_19 = None
        exp_19: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_20: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_466: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_80: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_466, [32, 16, 128, 128]);  convert_element_type_466 = None
        view_430: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_80, [512, 128, 128]);  expand_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_422: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [4096, 2048]);  convert_element_type_457 = None
        permute_211: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        mm_59: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_422, permute_211);  view_422 = permute_211 = None
        view_423: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [32, 128, 2048]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_426: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_423, [32, 128, 16, 128]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_214: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_81: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_214, [32, 16, 128, 128]);  permute_214 = None
        clone_137: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_431: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [512, 128, 128]);  clone_137 = None
        bmm_39: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_430, view_431);  view_430 = view_431 = None
        view_432: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [32, 16, 128, 128]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_216: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        clone_138: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_433: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [32, 128, 2048]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_434: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_433, [4096, 2048]);  view_433 = None
        permute_217: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_57: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg276_1, view_434, permute_217);  arg276_1 = view_434 = permute_217 = None
        view_435: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [32, 128, 2048]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_178: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_435, add_174);  view_435 = add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_472: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_178, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_472, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        sub_61: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_472, getitem_79);  convert_element_type_472 = getitem_79 = None
        add_179: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        mul_154: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_39);  sub_61 = rsqrt_39 = None
        mul_155: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, arg277_1);  mul_154 = arg277_1 = None
        add_180: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, arg278_1);  mul_155 = arg278_1 = None
        convert_element_type_473: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_180, torch.bfloat16);  add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_436: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_473, [4096, 2048]);  convert_element_type_473 = None
        permute_218: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_58: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg280_1, view_436, permute_218);  arg280_1 = view_436 = permute_218 = None
        view_437: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 128, 8192]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        pow_20: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_437, 3.0)
        mul_157: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_181: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_437, mul_157);  view_437 = mul_157 = None
        mul_158: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_181, 0.7978845608028654);  add_181 = None
        tanh_19: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_182: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_159: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, add_182);  mul_156 = add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_438: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_159, [4096, 8192]);  mul_159 = None
        permute_219: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_59: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg282_1, view_438, permute_219);  arg282_1 = view_438 = permute_219 = None
        view_439: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [32, 128, 2048]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_183: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_178, view_439);  add_178 = view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_480: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_183, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_480, [2], correction = 0, keepdim = True)
        getitem_80: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant21: "f32[][]cuda:0" = self._tensor_constant21;  _tensor_constant21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_44: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg288_1, 2, 0, 128);  arg288_1 = None
        slice_45: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_44, 3, 0, 128);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_62: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_480, getitem_81);  convert_element_type_480 = getitem_81 = None
        add_184: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        mul_160: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_40);  sub_62 = rsqrt_40 = None
        mul_161: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, arg283_1);  mul_160 = arg283_1 = None
        add_185: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, arg284_1);  mul_161 = arg284_1 = None
        convert_element_type_481: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_185, torch.bfloat16);  add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_440: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [4096, 2048])
        permute_220: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        mm_60: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_440, permute_220);  view_440 = permute_220 = None
        view_441: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [32, 128, 2048]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_446: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [32, 128, 16, 128]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_223: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_488: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_223, torch.float32);  permute_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_82: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_488, [32, 16, 128, 128]);  convert_element_type_488 = None
        clone_141: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_449: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_141, [512, 128, 128]);  clone_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_442: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [4096, 2048])
        permute_221: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        mm_61: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_442, permute_221);  view_442 = permute_221 = None
        view_443: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [32, 128, 2048]);  mm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_447: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_443, [32, 128, 16, 128]);  view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_224: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_489: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_224, torch.float32);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_226: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_489, [0, 1, 3, 2]);  convert_element_type_489 = None
        expand_83: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_226, [32, 16, 128, 128]);  permute_226 = None
        clone_142: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_450: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [512, 128, 128]);  clone_142 = None
        bmm_40: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_449, view_450);  view_449 = view_450 = None
        view_451: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [32, 16, 128, 128]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_22: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_45, view_451, full_default_22);  slice_45 = view_451 = full_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_186: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_21, where);  where_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_20: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_186, [-1], True)
        sub_63: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_186, amax_20);  add_186 = amax_20 = None
        exp_20: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
        sum_21: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_490: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_84: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_490, [32, 16, 128, 128]);  convert_element_type_490 = None
        view_452: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_84, [512, 128, 128]);  expand_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_444: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [4096, 2048]);  convert_element_type_481 = None
        permute_222: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        mm_62: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_444, permute_222);  view_444 = permute_222 = None
        view_445: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [32, 128, 2048]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_448: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_445, [32, 128, 16, 128]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_225: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_85: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_225, [32, 16, 128, 128]);  permute_225 = None
        clone_144: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_453: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_144, [512, 128, 128]);  clone_144 = None
        bmm_41: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_452, view_453);  view_452 = view_453 = None
        view_454: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [32, 16, 128, 128]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_227: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        clone_145: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_455: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [32, 128, 2048]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_456: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_455, [4096, 2048]);  view_455 = None
        permute_228: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_60: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg290_1, view_456, permute_228);  arg290_1 = view_456 = permute_228 = None
        view_457: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [32, 128, 2048]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_187: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_457, add_183);  view_457 = add_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_496: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_187, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_496, [2], correction = 0, keepdim = True)
        getitem_82: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        sub_64: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_496, getitem_83);  convert_element_type_496 = getitem_83 = None
        add_188: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_162: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_41);  sub_64 = rsqrt_41 = None
        mul_163: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, arg291_1);  mul_162 = arg291_1 = None
        add_189: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_163, arg292_1);  mul_163 = arg292_1 = None
        convert_element_type_497: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_458: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_497, [4096, 2048]);  convert_element_type_497 = None
        permute_229: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_61: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg294_1, view_458, permute_229);  arg294_1 = view_458 = permute_229 = None
        view_459: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 128, 8192]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_164: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_459, 0.5)
        pow_21: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_459, 3.0)
        mul_165: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_190: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_459, mul_165);  view_459 = mul_165 = None
        mul_166: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_190, 0.7978845608028654);  add_190 = None
        tanh_20: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_191: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_167: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, add_191);  mul_164 = add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_460: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_167, [4096, 8192]);  mul_167 = None
        permute_230: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_62: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg296_1, view_460, permute_230);  arg296_1 = view_460 = permute_230 = None
        view_461: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [32, 128, 2048]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_192: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_187, view_461);  add_187 = view_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_504: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_192, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_504, [2], correction = 0, keepdim = True)
        getitem_84: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant22: "f32[][]cuda:0" = self._tensor_constant22;  _tensor_constant22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_46: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg302_1, 2, 0, 128);  arg302_1 = None
        slice_47: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 128);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_65: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_504, getitem_85);  convert_element_type_504 = getitem_85 = None
        add_193: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
        mul_168: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_42);  sub_65 = rsqrt_42 = None
        mul_169: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, arg297_1);  mul_168 = arg297_1 = None
        add_194: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, arg298_1);  mul_169 = arg298_1 = None
        convert_element_type_505: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_194, torch.bfloat16);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_462: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [4096, 2048])
        permute_231: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        mm_63: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_462, permute_231);  view_462 = permute_231 = None
        view_463: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [32, 128, 2048]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_468: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_463, [32, 128, 16, 128]);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_234: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_512: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_234, torch.float32);  permute_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_86: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_512, [32, 16, 128, 128]);  convert_element_type_512 = None
        clone_148: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_471: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_148, [512, 128, 128]);  clone_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_464: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [4096, 2048])
        permute_232: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        mm_64: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_464, permute_232);  view_464 = permute_232 = None
        view_465: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [32, 128, 2048]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_469: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_465, [32, 128, 16, 128]);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_235: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_513: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_235, torch.float32);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_237: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_513, [0, 1, 3, 2]);  convert_element_type_513 = None
        expand_87: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_237, [32, 16, 128, 128]);  permute_237 = None
        clone_149: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_472: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_149, [512, 128, 128]);  clone_149 = None
        bmm_42: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_471, view_472);  view_471 = view_472 = None
        view_473: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [32, 16, 128, 128]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_23: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_47, view_473, full_default_23);  slice_47 = view_473 = full_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_195: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_22, where);  where_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_21: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_195, [-1], True)
        sub_66: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_195, amax_21);  add_195 = amax_21 = None
        exp_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
        sum_22: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_514: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_88: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_514, [32, 16, 128, 128]);  convert_element_type_514 = None
        view_474: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_88, [512, 128, 128]);  expand_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_466: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [4096, 2048]);  convert_element_type_505 = None
        permute_233: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        mm_65: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_466, permute_233);  view_466 = permute_233 = None
        view_467: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [32, 128, 2048]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_470: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_467, [32, 128, 16, 128]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_236: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_89: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_236, [32, 16, 128, 128]);  permute_236 = None
        clone_151: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_475: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [512, 128, 128]);  clone_151 = None
        bmm_43: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_474, view_475);  view_474 = view_475 = None
        view_476: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [32, 16, 128, 128]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_238: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_152: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_477: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_152, [32, 128, 2048]);  clone_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_478: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_477, [4096, 2048]);  view_477 = None
        permute_239: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_63: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg304_1, view_478, permute_239);  arg304_1 = view_478 = permute_239 = None
        view_479: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [32, 128, 2048]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_196: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_479, add_192);  view_479 = add_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_520: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_196, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_520, [2], correction = 0, keepdim = True)
        getitem_86: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        sub_67: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_520, getitem_87);  convert_element_type_520 = getitem_87 = None
        add_197: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        mul_170: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_43);  sub_67 = rsqrt_43 = None
        mul_171: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, arg305_1);  mul_170 = arg305_1 = None
        add_198: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_171, arg306_1);  mul_171 = arg306_1 = None
        convert_element_type_521: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16);  add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_480: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_521, [4096, 2048]);  convert_element_type_521 = None
        permute_240: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_64: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg308_1, view_480, permute_240);  arg308_1 = view_480 = permute_240 = None
        view_481: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 128, 8192]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_172: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        pow_22: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_481, 3.0)
        mul_173: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_199: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_481, mul_173);  view_481 = mul_173 = None
        mul_174: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_199, 0.7978845608028654);  add_199 = None
        tanh_21: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
        add_200: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_175: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, add_200);  mul_172 = add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_482: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_175, [4096, 8192]);  mul_175 = None
        permute_241: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_65: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg310_1, view_482, permute_241);  arg310_1 = view_482 = permute_241 = None
        view_483: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [32, 128, 2048]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_201: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_196, view_483);  add_196 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_528: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_201, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_528, [2], correction = 0, keepdim = True)
        getitem_88: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant23: "f32[][]cuda:0" = self._tensor_constant23;  _tensor_constant23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_48: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg316_1, 2, 0, 128);  arg316_1 = None
        slice_49: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_48, 3, 0, 128);  slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_68: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_528, getitem_89);  convert_element_type_528 = getitem_89 = None
        add_202: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        mul_176: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_44);  sub_68 = rsqrt_44 = None
        mul_177: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, arg311_1);  mul_176 = arg311_1 = None
        add_203: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_177, arg312_1);  mul_177 = arg312_1 = None
        convert_element_type_529: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_203, torch.bfloat16);  add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_484: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_529, [4096, 2048])
        permute_242: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        mm_66: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_484, permute_242);  view_484 = permute_242 = None
        view_485: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [32, 128, 2048]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_490: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [32, 128, 16, 128]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_245: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_536: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_245, torch.float32);  permute_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_90: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_536, [32, 16, 128, 128]);  convert_element_type_536 = None
        clone_155: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_493: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_155, [512, 128, 128]);  clone_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_486: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_529, [4096, 2048])
        permute_243: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
        mm_67: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_486, permute_243);  view_486 = permute_243 = None
        view_487: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [32, 128, 2048]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_491: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_487, [32, 128, 16, 128]);  view_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_246: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_537: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_246, torch.float32);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_248: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_537, [0, 1, 3, 2]);  convert_element_type_537 = None
        expand_91: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_248, [32, 16, 128, 128]);  permute_248 = None
        clone_156: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_494: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_156, [512, 128, 128]);  clone_156 = None
        bmm_44: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_493, view_494);  view_493 = view_494 = None
        view_495: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [32, 16, 128, 128]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_24: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_49, view_495, full_default_24);  slice_49 = view_495 = full_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_204: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_23, where);  where_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_22: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_204, [-1], True)
        sub_69: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_204, amax_22);  add_204 = amax_22 = None
        exp_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
        sum_23: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_538: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_92: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_538, [32, 16, 128, 128]);  convert_element_type_538 = None
        view_496: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_92, [512, 128, 128]);  expand_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_488: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_529, [4096, 2048]);  convert_element_type_529 = None
        permute_244: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        mm_68: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_488, permute_244);  view_488 = permute_244 = None
        view_489: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [32, 128, 2048]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_492: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_489, [32, 128, 16, 128]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_247: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_93: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_247, [32, 16, 128, 128]);  permute_247 = None
        clone_158: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_497: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_158, [512, 128, 128]);  clone_158 = None
        bmm_45: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_496, view_497);  view_496 = view_497 = None
        view_498: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [32, 16, 128, 128]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_249: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_159: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_499: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_159, [32, 128, 2048]);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_500: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_499, [4096, 2048]);  view_499 = None
        permute_250: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_66: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg318_1, view_500, permute_250);  arg318_1 = view_500 = permute_250 = None
        view_501: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [32, 128, 2048]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_205: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_501, add_201);  view_501 = add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_544: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_205, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_544, [2], correction = 0, keepdim = True)
        getitem_90: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        sub_70: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_544, getitem_91);  convert_element_type_544 = getitem_91 = None
        add_206: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        mul_178: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_45);  sub_70 = rsqrt_45 = None
        mul_179: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, arg319_1);  mul_178 = arg319_1 = None
        add_207: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, arg320_1);  mul_179 = arg320_1 = None
        convert_element_type_545: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_207, torch.bfloat16);  add_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_502: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_545, [4096, 2048]);  convert_element_type_545 = None
        permute_251: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_67: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg322_1, view_502, permute_251);  arg322_1 = view_502 = permute_251 = None
        view_503: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [32, 128, 8192]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_180: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        pow_23: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_503, 3.0)
        mul_181: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_208: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_503, mul_181);  view_503 = mul_181 = None
        mul_182: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_208, 0.7978845608028654);  add_208 = None
        tanh_22: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
        add_209: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_183: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, add_209);  mul_180 = add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_504: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_183, [4096, 8192]);  mul_183 = None
        permute_252: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_68: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg324_1, view_504, permute_252);  arg324_1 = view_504 = permute_252 = None
        view_505: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [32, 128, 2048]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_210: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_205, view_505);  add_205 = view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_552: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_210, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_552, [2], correction = 0, keepdim = True)
        getitem_92: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_46[0]
        getitem_93: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        _tensor_constant24: "f32[][]cuda:0" = self._tensor_constant24;  _tensor_constant24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_50: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg330_1, 2, 0, 128);  arg330_1 = None
        slice_51: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_50, 3, 0, 128);  slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_71: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_552, getitem_93);  convert_element_type_552 = getitem_93 = None
        add_211: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        mul_184: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_46);  sub_71 = rsqrt_46 = None
        mul_185: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, arg325_1);  mul_184 = arg325_1 = None
        add_212: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_185, arg326_1);  mul_185 = arg326_1 = None
        convert_element_type_553: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_212, torch.bfloat16);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_506: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_553, [4096, 2048])
        permute_253: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        mm_69: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_506, permute_253);  view_506 = permute_253 = None
        view_507: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [32, 128, 2048]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_512: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_507, [32, 128, 16, 128]);  view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_256: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_560: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_256, torch.float32);  permute_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_94: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_560, [32, 16, 128, 128]);  convert_element_type_560 = None
        clone_162: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_515: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [512, 128, 128]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        view_508: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_553, [4096, 2048])
        permute_254: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        mm_70: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_508, permute_254);  view_508 = permute_254 = None
        view_509: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [32, 128, 2048]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_513: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_509, [32, 128, 16, 128]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_257: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_561: "f32[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_257, torch.float32);  permute_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_259: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_561, [0, 1, 3, 2]);  convert_element_type_561 = None
        expand_95: "f32[32, 16, 128, 128][262144, 128, 1, 2048]cuda:0" = torch.ops.aten.expand.default(permute_259, [32, 16, 128, 128]);  permute_259 = None
        clone_163: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_516: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_163, [512, 128, 128]);  clone_163 = None
        bmm_46: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_515, view_516);  view_515 = view_516 = None
        view_517: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [32, 16, 128, 128]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_25: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_24: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_51, view_517, full_default_25);  slice_51 = view_517 = full_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_213: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_24, where);  where_24 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_23: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_213, [-1], True)
        sub_72: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_213, amax_23);  add_213 = amax_23 = None
        exp_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_72);  sub_72 = None
        sum_24: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_562: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_96: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_562, [32, 16, 128, 128]);  convert_element_type_562 = None
        view_518: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_96, [512, 128, 128]);  expand_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_510: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_553, [4096, 2048]);  convert_element_type_553 = None
        permute_255: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        mm_71: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_510, permute_255);  view_510 = permute_255 = None
        view_511: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [32, 128, 2048]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_514: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [32, 128, 16, 128]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_258: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_97: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.expand.default(permute_258, [32, 16, 128, 128]);  permute_258 = None
        clone_165: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_519: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_165, [512, 128, 128]);  clone_165 = None
        bmm_47: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_518, view_519);  view_518 = view_519 = None
        view_520: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [32, 16, 128, 128]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_260: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
        clone_166: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_521: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_166, [32, 128, 2048]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_522: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_521, [4096, 2048]);  view_521 = None
        permute_261: "bf16[2048, 2048][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_69: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg332_1, view_522, permute_261);  arg332_1 = view_522 = permute_261 = None
        view_523: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [32, 128, 2048]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_214: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_523, add_210);  view_523 = add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_568: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_568, [2], correction = 0, keepdim = True)
        getitem_94: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_47[0]
        getitem_95: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        sub_73: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_568, getitem_95);  convert_element_type_568 = getitem_95 = None
        add_215: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        mul_186: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_47);  sub_73 = rsqrt_47 = None
        mul_187: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, arg333_1);  mul_186 = arg333_1 = None
        add_216: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_187, arg334_1);  mul_187 = arg334_1 = None
        convert_element_type_569: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_216, torch.bfloat16);  add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_524: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_569, [4096, 2048]);  convert_element_type_569 = None
        permute_262: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_70: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.addmm.default(arg336_1, view_524, permute_262);  arg336_1 = view_524 = permute_262 = None
        view_525: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 128, 8192]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_188: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        pow_24: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_525, 3.0)
        mul_189: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_217: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_525, mul_189);  view_525 = mul_189 = None
        mul_190: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_217, 0.7978845608028654);  add_217 = None
        tanh_23: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_218: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_191: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_188, add_218);  mul_188 = add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_526: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_191, [4096, 8192]);  mul_191 = None
        permute_263: "bf16[8192, 2048][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_71: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(arg338_1, view_526, permute_263);  arg338_1 = view_526 = permute_263 = None
        view_527: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [32, 128, 2048]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_219: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_214, view_527);  add_214 = view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:492 in forward, code: hidden_states = self.ln_f(hidden_states)
        convert_element_type_576: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.float32);  add_219 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_576, [2], correction = 0, keepdim = True)
        getitem_96: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_48[0]
        getitem_97: "f32[32, 128, 1][128, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[32, 129][129, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(arg341_1, [0, 1], -100.0);  arg341_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_52: "i64[32, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_169: "i64[32, 128][128, 1]cuda:0" = torch.ops.aten.clone.default(slice_52, memory_format = torch.contiguous_format);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_532: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [-1]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_2: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_532, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:492 in forward, code: hidden_states = self.ln_f(hidden_states)
        sub_74: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_576, getitem_97);  convert_element_type_576 = getitem_97 = None
        add_220: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        mul_192: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_48);  sub_74 = rsqrt_48 = None
        mul_193: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, arg339_1);  mul_192 = arg339_1 = None
        add_221: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, arg340_1);  mul_193 = arg340_1 = None
        convert_element_type_577: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_529: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_577, [4096, 2048]);  convert_element_type_577 = None
        permute_264: "bf16[2048, 50257][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "bf16[2048, 50264][50264, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_264, [0, 7, 0, 0]);  permute_264 = None
        mm_default: "bf16[4096, 50264][50264, 1]cuda:0" = torch.ops.aten.mm.default(view_529, constant_pad_nd_default);  view_529 = constant_pad_nd_default = None
        slice_tensor: "bf16[4096, 50257][50264, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -7);  mm_default = None
        view_530: "bf16[32, 128, 50257][6433792, 50264, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [32, 128, 50257]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_580: "f32[32, 128, 50257][6432896, 50257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_530, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_531: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_580, [-1, 50257]);  convert_element_type_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_24: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(view_531, [1], True)
        sub_75: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_531, amax_24);  view_531 = amax_24 = None
        exp_24: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.exp.default(sub_75)
        sum_25: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_76: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = log = None
        ne_1: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_532, -100)
        full_default_26: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "i64[4096][1]cuda:0" = torch.ops.aten.where.self(ne_1, view_532, full_default_26);  ne_1 = full_default_26 = None
        unsqueeze_10: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_25, 1);  where_25 = None
        gather: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(sub_76, 1, unsqueeze_10);  sub_76 = unsqueeze_10 = None
        squeeze: "f32[4096][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_27: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "f32[4096][1]cuda:0" = torch.ops.aten.where.self(ne_2, neg, full_default_27);  ne_2 = neg = full_default_27 = None
        sum_27: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_26);  where_26 = None
        ne_3: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_532, -100);  view_532 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_581: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        div_24: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_581);  sum_27 = convert_element_type_581 = None
        return (div_24, view_530)

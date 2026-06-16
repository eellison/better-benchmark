class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 1000][1000, 1]cuda:0", arg1_1: "bf16[256000, 2304][2304, 1]cuda:0", arg2_1: "bf16[][]cuda:0", arg3_1: "bf16[128][1]cuda:0", arg4_1: "bf16[2304][1]cuda:0", arg5_1: "bf16[2048, 2304][2304, 1]cuda:0", arg6_1: "bf16[1024, 2304][2304, 1]cuda:0", arg7_1: "bf16[1024, 2304][2304, 1]cuda:0", arg8_1: "bf16[2304, 2048][2048, 1]cuda:0", arg9_1: "bf16[2304][1]cuda:0", arg10_1: "bf16[2304][1]cuda:0", arg11_1: "bf16[9216, 2304][2304, 1]cuda:0", arg12_1: "bf16[9216, 2304][2304, 1]cuda:0", arg13_1: "bf16[2304, 9216][9216, 1]cuda:0", arg14_1: "bf16[2304][1]cuda:0", arg15_1: "bf16[2304][1]cuda:0", arg16_1: "bf16[2048, 2304][2304, 1]cuda:0", arg17_1: "bf16[1024, 2304][2304, 1]cuda:0", arg18_1: "bf16[1024, 2304][2304, 1]cuda:0", arg19_1: "bf16[2304, 2048][2048, 1]cuda:0", arg20_1: "bf16[2304][1]cuda:0", arg21_1: "bf16[2304][1]cuda:0", arg22_1: "bf16[9216, 2304][2304, 1]cuda:0", arg23_1: "bf16[9216, 2304][2304, 1]cuda:0", arg24_1: "bf16[2304, 9216][9216, 1]cuda:0", arg25_1: "bf16[2304][1]cuda:0", arg26_1: "bf16[2304][1]cuda:0", arg27_1: "bf16[2048, 2304][2304, 1]cuda:0", arg28_1: "bf16[1024, 2304][2304, 1]cuda:0", arg29_1: "bf16[1024, 2304][2304, 1]cuda:0", arg30_1: "bf16[2304, 2048][2048, 1]cuda:0", arg31_1: "bf16[2304][1]cuda:0", arg32_1: "bf16[2304][1]cuda:0", arg33_1: "bf16[9216, 2304][2304, 1]cuda:0", arg34_1: "bf16[9216, 2304][2304, 1]cuda:0", arg35_1: "bf16[2304, 9216][9216, 1]cuda:0", arg36_1: "bf16[2304][1]cuda:0", arg37_1: "bf16[2304][1]cuda:0", arg38_1: "bf16[2048, 2304][2304, 1]cuda:0", arg39_1: "bf16[1024, 2304][2304, 1]cuda:0", arg40_1: "bf16[1024, 2304][2304, 1]cuda:0", arg41_1: "bf16[2304, 2048][2048, 1]cuda:0", arg42_1: "bf16[2304][1]cuda:0", arg43_1: "bf16[2304][1]cuda:0", arg44_1: "bf16[9216, 2304][2304, 1]cuda:0", arg45_1: "bf16[9216, 2304][2304, 1]cuda:0", arg46_1: "bf16[2304, 9216][9216, 1]cuda:0", arg47_1: "bf16[2304][1]cuda:0", arg48_1: "bf16[2304][1]cuda:0", arg49_1: "bf16[2048, 2304][2304, 1]cuda:0", arg50_1: "bf16[1024, 2304][2304, 1]cuda:0", arg51_1: "bf16[1024, 2304][2304, 1]cuda:0", arg52_1: "bf16[2304, 2048][2048, 1]cuda:0", arg53_1: "bf16[2304][1]cuda:0", arg54_1: "bf16[2304][1]cuda:0", arg55_1: "bf16[9216, 2304][2304, 1]cuda:0", arg56_1: "bf16[9216, 2304][2304, 1]cuda:0", arg57_1: "bf16[2304, 9216][9216, 1]cuda:0", arg58_1: "bf16[2304][1]cuda:0", arg59_1: "bf16[2304][1]cuda:0", arg60_1: "bf16[2048, 2304][2304, 1]cuda:0", arg61_1: "bf16[1024, 2304][2304, 1]cuda:0", arg62_1: "bf16[1024, 2304][2304, 1]cuda:0", arg63_1: "bf16[2304, 2048][2048, 1]cuda:0", arg64_1: "bf16[2304][1]cuda:0", arg65_1: "bf16[2304][1]cuda:0", arg66_1: "bf16[9216, 2304][2304, 1]cuda:0", arg67_1: "bf16[9216, 2304][2304, 1]cuda:0", arg68_1: "bf16[2304, 9216][9216, 1]cuda:0", arg69_1: "bf16[2304][1]cuda:0", arg70_1: "bf16[2304][1]cuda:0", arg71_1: "bf16[2048, 2304][2304, 1]cuda:0", arg72_1: "bf16[1024, 2304][2304, 1]cuda:0", arg73_1: "bf16[1024, 2304][2304, 1]cuda:0", arg74_1: "bf16[2304, 2048][2048, 1]cuda:0", arg75_1: "bf16[2304][1]cuda:0", arg76_1: "bf16[2304][1]cuda:0", arg77_1: "bf16[9216, 2304][2304, 1]cuda:0", arg78_1: "bf16[9216, 2304][2304, 1]cuda:0", arg79_1: "bf16[2304, 9216][9216, 1]cuda:0", arg80_1: "bf16[2304][1]cuda:0", arg81_1: "bf16[2304][1]cuda:0", arg82_1: "bf16[2048, 2304][2304, 1]cuda:0", arg83_1: "bf16[1024, 2304][2304, 1]cuda:0", arg84_1: "bf16[1024, 2304][2304, 1]cuda:0", arg85_1: "bf16[2304, 2048][2048, 1]cuda:0", arg86_1: "bf16[2304][1]cuda:0", arg87_1: "bf16[2304][1]cuda:0", arg88_1: "bf16[9216, 2304][2304, 1]cuda:0", arg89_1: "bf16[9216, 2304][2304, 1]cuda:0", arg90_1: "bf16[2304, 9216][9216, 1]cuda:0", arg91_1: "bf16[2304][1]cuda:0", arg92_1: "bf16[2304][1]cuda:0", arg93_1: "bf16[2048, 2304][2304, 1]cuda:0", arg94_1: "bf16[1024, 2304][2304, 1]cuda:0", arg95_1: "bf16[1024, 2304][2304, 1]cuda:0", arg96_1: "bf16[2304, 2048][2048, 1]cuda:0", arg97_1: "bf16[2304][1]cuda:0", arg98_1: "bf16[2304][1]cuda:0", arg99_1: "bf16[9216, 2304][2304, 1]cuda:0", arg100_1: "bf16[9216, 2304][2304, 1]cuda:0", arg101_1: "bf16[2304, 9216][9216, 1]cuda:0", arg102_1: "bf16[2304][1]cuda:0", arg103_1: "bf16[2304][1]cuda:0", arg104_1: "bf16[2048, 2304][2304, 1]cuda:0", arg105_1: "bf16[1024, 2304][2304, 1]cuda:0", arg106_1: "bf16[1024, 2304][2304, 1]cuda:0", arg107_1: "bf16[2304, 2048][2048, 1]cuda:0", arg108_1: "bf16[2304][1]cuda:0", arg109_1: "bf16[2304][1]cuda:0", arg110_1: "bf16[9216, 2304][2304, 1]cuda:0", arg111_1: "bf16[9216, 2304][2304, 1]cuda:0", arg112_1: "bf16[2304, 9216][9216, 1]cuda:0", arg113_1: "bf16[2304][1]cuda:0", arg114_1: "bf16[2304][1]cuda:0", arg115_1: "bf16[2048, 2304][2304, 1]cuda:0", arg116_1: "bf16[1024, 2304][2304, 1]cuda:0", arg117_1: "bf16[1024, 2304][2304, 1]cuda:0", arg118_1: "bf16[2304, 2048][2048, 1]cuda:0", arg119_1: "bf16[2304][1]cuda:0", arg120_1: "bf16[2304][1]cuda:0", arg121_1: "bf16[9216, 2304][2304, 1]cuda:0", arg122_1: "bf16[9216, 2304][2304, 1]cuda:0", arg123_1: "bf16[2304, 9216][9216, 1]cuda:0", arg124_1: "bf16[2304][1]cuda:0", arg125_1: "bf16[2304][1]cuda:0", arg126_1: "bf16[2048, 2304][2304, 1]cuda:0", arg127_1: "bf16[1024, 2304][2304, 1]cuda:0", arg128_1: "bf16[1024, 2304][2304, 1]cuda:0", arg129_1: "bf16[2304, 2048][2048, 1]cuda:0", arg130_1: "bf16[2304][1]cuda:0", arg131_1: "bf16[2304][1]cuda:0", arg132_1: "bf16[9216, 2304][2304, 1]cuda:0", arg133_1: "bf16[9216, 2304][2304, 1]cuda:0", arg134_1: "bf16[2304, 9216][9216, 1]cuda:0", arg135_1: "bf16[2304][1]cuda:0", arg136_1: "bf16[2304][1]cuda:0", arg137_1: "bf16[2048, 2304][2304, 1]cuda:0", arg138_1: "bf16[1024, 2304][2304, 1]cuda:0", arg139_1: "bf16[1024, 2304][2304, 1]cuda:0", arg140_1: "bf16[2304, 2048][2048, 1]cuda:0", arg141_1: "bf16[2304][1]cuda:0", arg142_1: "bf16[2304][1]cuda:0", arg143_1: "bf16[9216, 2304][2304, 1]cuda:0", arg144_1: "bf16[9216, 2304][2304, 1]cuda:0", arg145_1: "bf16[2304, 9216][9216, 1]cuda:0", arg146_1: "bf16[2304][1]cuda:0", arg147_1: "bf16[2304][1]cuda:0", arg148_1: "bf16[2048, 2304][2304, 1]cuda:0", arg149_1: "bf16[1024, 2304][2304, 1]cuda:0", arg150_1: "bf16[1024, 2304][2304, 1]cuda:0", arg151_1: "bf16[2304, 2048][2048, 1]cuda:0", arg152_1: "bf16[2304][1]cuda:0", arg153_1: "bf16[2304][1]cuda:0", arg154_1: "bf16[9216, 2304][2304, 1]cuda:0", arg155_1: "bf16[9216, 2304][2304, 1]cuda:0", arg156_1: "bf16[2304, 9216][9216, 1]cuda:0", arg157_1: "bf16[2304][1]cuda:0", arg158_1: "bf16[2304][1]cuda:0", arg159_1: "bf16[2048, 2304][2304, 1]cuda:0", arg160_1: "bf16[1024, 2304][2304, 1]cuda:0", arg161_1: "bf16[1024, 2304][2304, 1]cuda:0", arg162_1: "bf16[2304, 2048][2048, 1]cuda:0", arg163_1: "bf16[2304][1]cuda:0", arg164_1: "bf16[2304][1]cuda:0", arg165_1: "bf16[9216, 2304][2304, 1]cuda:0", arg166_1: "bf16[9216, 2304][2304, 1]cuda:0", arg167_1: "bf16[2304, 9216][9216, 1]cuda:0", arg168_1: "bf16[2304][1]cuda:0", arg169_1: "bf16[2304][1]cuda:0", arg170_1: "bf16[2048, 2304][2304, 1]cuda:0", arg171_1: "bf16[1024, 2304][2304, 1]cuda:0", arg172_1: "bf16[1024, 2304][2304, 1]cuda:0", arg173_1: "bf16[2304, 2048][2048, 1]cuda:0", arg174_1: "bf16[2304][1]cuda:0", arg175_1: "bf16[2304][1]cuda:0", arg176_1: "bf16[9216, 2304][2304, 1]cuda:0", arg177_1: "bf16[9216, 2304][2304, 1]cuda:0", arg178_1: "bf16[2304, 9216][9216, 1]cuda:0", arg179_1: "bf16[2304][1]cuda:0", arg180_1: "bf16[2304][1]cuda:0", arg181_1: "bf16[2048, 2304][2304, 1]cuda:0", arg182_1: "bf16[1024, 2304][2304, 1]cuda:0", arg183_1: "bf16[1024, 2304][2304, 1]cuda:0", arg184_1: "bf16[2304, 2048][2048, 1]cuda:0", arg185_1: "bf16[2304][1]cuda:0", arg186_1: "bf16[2304][1]cuda:0", arg187_1: "bf16[9216, 2304][2304, 1]cuda:0", arg188_1: "bf16[9216, 2304][2304, 1]cuda:0", arg189_1: "bf16[2304, 9216][9216, 1]cuda:0", arg190_1: "bf16[2304][1]cuda:0", arg191_1: "bf16[2304][1]cuda:0", arg192_1: "bf16[2048, 2304][2304, 1]cuda:0", arg193_1: "bf16[1024, 2304][2304, 1]cuda:0", arg194_1: "bf16[1024, 2304][2304, 1]cuda:0", arg195_1: "bf16[2304, 2048][2048, 1]cuda:0", arg196_1: "bf16[2304][1]cuda:0", arg197_1: "bf16[2304][1]cuda:0", arg198_1: "bf16[9216, 2304][2304, 1]cuda:0", arg199_1: "bf16[9216, 2304][2304, 1]cuda:0", arg200_1: "bf16[2304, 9216][9216, 1]cuda:0", arg201_1: "bf16[2304][1]cuda:0", arg202_1: "bf16[2304][1]cuda:0", arg203_1: "bf16[2048, 2304][2304, 1]cuda:0", arg204_1: "bf16[1024, 2304][2304, 1]cuda:0", arg205_1: "bf16[1024, 2304][2304, 1]cuda:0", arg206_1: "bf16[2304, 2048][2048, 1]cuda:0", arg207_1: "bf16[2304][1]cuda:0", arg208_1: "bf16[2304][1]cuda:0", arg209_1: "bf16[9216, 2304][2304, 1]cuda:0", arg210_1: "bf16[9216, 2304][2304, 1]cuda:0", arg211_1: "bf16[2304, 9216][9216, 1]cuda:0", arg212_1: "bf16[2304][1]cuda:0", arg213_1: "bf16[2304][1]cuda:0", arg214_1: "bf16[2048, 2304][2304, 1]cuda:0", arg215_1: "bf16[1024, 2304][2304, 1]cuda:0", arg216_1: "bf16[1024, 2304][2304, 1]cuda:0", arg217_1: "bf16[2304, 2048][2048, 1]cuda:0", arg218_1: "bf16[2304][1]cuda:0", arg219_1: "bf16[2304][1]cuda:0", arg220_1: "bf16[9216, 2304][2304, 1]cuda:0", arg221_1: "bf16[9216, 2304][2304, 1]cuda:0", arg222_1: "bf16[2304, 9216][9216, 1]cuda:0", arg223_1: "bf16[2304][1]cuda:0", arg224_1: "bf16[2304][1]cuda:0", arg225_1: "bf16[2048, 2304][2304, 1]cuda:0", arg226_1: "bf16[1024, 2304][2304, 1]cuda:0", arg227_1: "bf16[1024, 2304][2304, 1]cuda:0", arg228_1: "bf16[2304, 2048][2048, 1]cuda:0", arg229_1: "bf16[2304][1]cuda:0", arg230_1: "bf16[2304][1]cuda:0", arg231_1: "bf16[9216, 2304][2304, 1]cuda:0", arg232_1: "bf16[9216, 2304][2304, 1]cuda:0", arg233_1: "bf16[2304, 9216][9216, 1]cuda:0", arg234_1: "bf16[2304][1]cuda:0", arg235_1: "bf16[2304][1]cuda:0", arg236_1: "bf16[2048, 2304][2304, 1]cuda:0", arg237_1: "bf16[1024, 2304][2304, 1]cuda:0", arg238_1: "bf16[1024, 2304][2304, 1]cuda:0", arg239_1: "bf16[2304, 2048][2048, 1]cuda:0", arg240_1: "bf16[2304][1]cuda:0", arg241_1: "bf16[2304][1]cuda:0", arg242_1: "bf16[9216, 2304][2304, 1]cuda:0", arg243_1: "bf16[9216, 2304][2304, 1]cuda:0", arg244_1: "bf16[2304, 9216][9216, 1]cuda:0", arg245_1: "bf16[2304][1]cuda:0", arg246_1: "bf16[2304][1]cuda:0", arg247_1: "bf16[2048, 2304][2304, 1]cuda:0", arg248_1: "bf16[1024, 2304][2304, 1]cuda:0", arg249_1: "bf16[1024, 2304][2304, 1]cuda:0", arg250_1: "bf16[2304, 2048][2048, 1]cuda:0", arg251_1: "bf16[2304][1]cuda:0", arg252_1: "bf16[2304][1]cuda:0", arg253_1: "bf16[9216, 2304][2304, 1]cuda:0", arg254_1: "bf16[9216, 2304][2304, 1]cuda:0", arg255_1: "bf16[2304, 9216][9216, 1]cuda:0", arg256_1: "bf16[2304][1]cuda:0", arg257_1: "bf16[2304][1]cuda:0", arg258_1: "bf16[2048, 2304][2304, 1]cuda:0", arg259_1: "bf16[1024, 2304][2304, 1]cuda:0", arg260_1: "bf16[1024, 2304][2304, 1]cuda:0", arg261_1: "bf16[2304, 2048][2048, 1]cuda:0", arg262_1: "bf16[2304][1]cuda:0", arg263_1: "bf16[2304][1]cuda:0", arg264_1: "bf16[9216, 2304][2304, 1]cuda:0", arg265_1: "bf16[9216, 2304][2304, 1]cuda:0", arg266_1: "bf16[2304, 9216][9216, 1]cuda:0", arg267_1: "bf16[2304][1]cuda:0", arg268_1: "bf16[2304][1]cuda:0", arg269_1: "bf16[2048, 2304][2304, 1]cuda:0", arg270_1: "bf16[1024, 2304][2304, 1]cuda:0", arg271_1: "bf16[1024, 2304][2304, 1]cuda:0", arg272_1: "bf16[2304, 2048][2048, 1]cuda:0", arg273_1: "bf16[2304][1]cuda:0", arg274_1: "bf16[2304][1]cuda:0", arg275_1: "bf16[9216, 2304][2304, 1]cuda:0", arg276_1: "bf16[9216, 2304][2304, 1]cuda:0", arg277_1: "bf16[2304, 9216][9216, 1]cuda:0", arg278_1: "bf16[2304][1]cuda:0", arg279_1: "bf16[2304][1]cuda:0", arg280_1: "bf16[2048, 2304][2304, 1]cuda:0", arg281_1: "bf16[1024, 2304][2304, 1]cuda:0", arg282_1: "bf16[1024, 2304][2304, 1]cuda:0", arg283_1: "bf16[2304, 2048][2048, 1]cuda:0", arg284_1: "bf16[2304][1]cuda:0", arg285_1: "bf16[2304][1]cuda:0", arg286_1: "bf16[9216, 2304][2304, 1]cuda:0", arg287_1: "bf16[9216, 2304][2304, 1]cuda:0", arg288_1: "bf16[2304, 9216][9216, 1]cuda:0", arg289_1: "bf16[2304][1]cuda:0", arg290_1: "bf16[2304][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:208 in __init__, code: self._sliding_window_tensor = torch.tensor(self.sliding_window, dtype=torch.long)
        _tensor_constant0: "i64[][]cpu" = self._tensor_constant0
        lift_fresh_copy: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = lift_fresh_copy = None
        _tensor_constant1: "i64[][]cpu" = self._tensor_constant1
        lift_fresh_copy_1: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = lift_fresh_copy_1 = None
        _tensor_constant2: "i64[][]cpu" = self._tensor_constant2
        lift_fresh_copy_2: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = lift_fresh_copy_2 = None
        _tensor_constant3: "i64[][]cpu" = self._tensor_constant3
        lift_fresh_copy_3: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = lift_fresh_copy_3 = None
        _tensor_constant4: "i64[][]cpu" = self._tensor_constant4
        lift_fresh_copy_4: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = lift_fresh_copy_4 = None
        _tensor_constant5: "i64[][]cpu" = self._tensor_constant5
        lift_fresh_copy_5: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = lift_fresh_copy_5 = None
        _tensor_constant6: "i64[][]cpu" = self._tensor_constant6
        lift_fresh_copy_6: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = lift_fresh_copy_6 = None
        _tensor_constant7: "i64[][]cpu" = self._tensor_constant7
        lift_fresh_copy_7: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = lift_fresh_copy_7 = None
        _tensor_constant8: "i64[][]cpu" = self._tensor_constant8
        lift_fresh_copy_8: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = lift_fresh_copy_8 = None
        _tensor_constant9: "i64[][]cpu" = self._tensor_constant9
        lift_fresh_copy_9: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = lift_fresh_copy_9 = None
        _tensor_constant10: "i64[][]cpu" = self._tensor_constant10
        lift_fresh_copy_10: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = lift_fresh_copy_10 = None
        _tensor_constant11: "i64[][]cpu" = self._tensor_constant11
        lift_fresh_copy_11: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = lift_fresh_copy_11 = None
        _tensor_constant12: "i64[][]cpu" = self._tensor_constant12
        lift_fresh_copy_12: "i64[][]cpu" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = lift_fresh_copy_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:360 in forward, code: return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
        embedding: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None
        mul: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, arg2_1);  embedding = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_4: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_1: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 2)
        mean: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_5: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_4: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, rsqrt);  convert_element_type_4 = rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_5: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        add_6: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_5, 1.0);  convert_element_type_5 = None
        mul_5: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_6: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_4: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_6, [1000, 2304])
        permute_1: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
        view_5: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [1, 1000, 2048]);  mm = None
        view_6: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [1, 1000, -1, 256]);  view_5 = None
        permute_2: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:137 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_13: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg3_1, 0);  arg3_1 = None
        unsqueeze_14: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        convert_element_type: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_14, torch.float32);  unsqueeze_14 = None
        expand_2: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type, [1, -1, 1]);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:142 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.expand.default(expand_2, [1, 128, 1]);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:435 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[1000][1]cuda:0" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1000][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:436 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 1000][1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:138 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_15: "i64[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        convert_element_type_1: "f32[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_15, torch.float32);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:142 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_4: "f32[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_1, [1, 1, 1000]);  convert_element_type_1 = None
        mul_1: "f32[1, 128, 1000][128000, 1000, 1]cuda:0" = torch.ops.aten.mul.Tensor(expand_3, expand_4);  expand_3 = expand_4 = None
        permute: "f32[1, 1000, 128][128000, 1, 1000]cuda:0" = torch.ops.aten.permute.default(mul_1, [0, 2, 1]);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:143 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_16: "f32[1, 1000, 1, 128][128000, 1, 128000, 1000]cuda:0" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_5: "f32[1, 1000, 2, 128][128000, 1, 0, 1000]cuda:0" = torch.ops.aten.expand.default(unsqueeze_16, [1, 1000, 2, 128]);  unsqueeze_16 = None
        clone: "f32[1, 1000, 2, 128][256000, 256, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_3: "f32[1, 1000, 256][256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [1, 1000, 256]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:144 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 1000, 256][256000, 256, 1]cuda:0" = torch.ops.aten.cos.default(view_3)
        mul_2: "f32[1, 1000, 256][256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:147 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 1000, 256][256000, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_17: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_6: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_2, unsqueeze_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_2: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_2, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_2);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_1: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 128);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg, slice_1], -1);  neg = slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:145 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 1000, 256][256000, 256, 1]cuda:0" = torch.ops.aten.sin.default(view_3);  view_3 = None
        mul_3: "f32[1, 1000, 256][256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:147 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_3: "bf16[1, 1000, 256][256000, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_18: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_7: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat, unsqueeze_18);  cat = None
        add_7: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_7: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_6, [1000, 2304])
        permute_3: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_1: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
        view_8: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [1, 1000, 1024]);  mm_1 = None
        view_9: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_8, [1, 1000, -1, 256]);  view_8 = None
        permute_4: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_8: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_17);  unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_4: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_4, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_4);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_3: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 128);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_1: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_1, slice_3], -1);  neg_1 = slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_9: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_18);  cat_1 = unsqueeze_18 = None
        add_8: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_19: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_8, 2)
        expand_6: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_19, [1, 4, 2, 1000, 256]);  unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_4: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_13: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [1, 8, 1000, 256]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_10: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_6, [1000, 2304]);  convert_element_type_6 = None
        permute_5: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_2: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [1, 1000, 1024]);  mm_2 = None
        view_12: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_11, [1, 1000, -1, 256]);  view_11 = None
        permute_6: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_20: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_6, 2)
        expand_7: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_20, [1, 4, 2, 1000, 256]);  unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_5: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_14: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [1, 8, 1000, 256]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_8: "i64[1000][1]cuda:0" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[1000][1]cuda:0" = torch.ops.aten.add.Tensor(iota_8, 0);  iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_10: "i64[1, 1000][1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_11: "i64[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
        unsqueeze_12: "i64[1, 1, 1, 1000][1000, 1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 2);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_7: "i64[1000][1]cuda:0" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[1000][1]cuda:0" = torch.ops.aten.add.Tensor(iota_7, 0);  iota_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_7: "i64[1, 1000][1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_8: "i64[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1000, 1][1000, 1000, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:97 in inner_mask, code: return kv_idx > q_idx - sliding_window
        sub: "i64[1, 1, 1000, 1][1000, 1000, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_9, 4096)
        gt: "b8[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.gt.Tensor(unsqueeze_12, sub);  sub = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full, gt);  full = gt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le_1: "b8[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_12, unsqueeze_9);  unsqueeze_12 = unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, le_1);  bitwise_and = le_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_1, [1, -1, 1000, 1000]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        expand_8: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where, [1, 8, 1000, 1000]);  where = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_7, view_13, view_14, expand_8, False, scale = 0.0625);  add_7 = view_13 = view_14 = expand_8 = None
        getitem: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_7, [1, 1000, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_16: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [1000, 2048]);  view_15 = None
        permute_8: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        mm_3: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
        view_17: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [1, 1000, 2304]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_16: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_2: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_16, 2)
        mean_1: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_9: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_10: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_16, rsqrt_1);  convert_element_type_16 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_17: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        add_10: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_17, 1.0);  convert_element_type_17 = None
        mul_11: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, add_10);  mul_10 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_18: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_11: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, convert_element_type_18);  mul = convert_element_type_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_19: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_3: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_19, 2)
        mean_2: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_12: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_12: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, rsqrt_2);  convert_element_type_19 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_20: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32);  arg10_1 = None
        add_13: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_20, 1.0);  convert_element_type_20 = None
        mul_13: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_13);  mul_12 = add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_21: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_18: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_21, [1000, 2304])
        permute_9: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_4: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
        view_19: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [1, 1000, 9216]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_24: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_18: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5)
        mul_14: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, convert_element_type_24)
        mul_15: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, convert_element_type_24);  mul_14 = None
        mul_16: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, 0.044715);  mul_15 = None
        add_14: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_24, mul_16);  convert_element_type_24 = mul_16 = None
        mul_17: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_17);  mul_17 = None
        add_15: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_19: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, add_15);  mul_18 = add_15 = None
        convert_element_type_25: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_20: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_21, [1000, 2304]);  convert_element_type_21 = None
        permute_10: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm_5: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
        view_21: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [1, 1000, 9216]);  mm_5 = None
        mul_20: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, view_21);  convert_element_type_25 = view_21 = None
        view_22: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_20, [1000, 9216]);  mul_20 = None
        permute_11: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_6: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
        view_23: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [1, 1000, 2304]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_30: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.float32);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_4: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_30, 2)
        mean_3: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_16: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_21: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_30, rsqrt_3);  convert_element_type_30 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_31: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float32);  arg14_1 = None
        add_17: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_31, 1.0);  convert_element_type_31 = None
        mul_22: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, add_17);  mul_21 = add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_32: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_18: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, convert_element_type_32);  add_11 = convert_element_type_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_33: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_5: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_33, 2)
        mean_4: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_19: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_23: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_33, rsqrt_4);  convert_element_type_33 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_34: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg15_1, torch.float32);  arg15_1 = None
        add_20: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_34, 1.0);  convert_element_type_34 = None
        mul_24: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, add_20);  mul_23 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_35: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_24: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [1000, 2304])
        permute_12: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_7: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
        view_25: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [1, 1000, 2048]);  mm_7 = None
        view_26: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [1, 1000, -1, 256]);  view_25 = None
        permute_13: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_21: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_25: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_13, unsqueeze_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_8: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_13, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_2: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_8);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_7: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_13, 3, 0, 128);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_2: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_2, slice_7], -1);  neg_2 = slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_22: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_26: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_22);  cat_2 = None
        add_21: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_27: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [1000, 2304])
        permute_14: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_8: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_27, permute_14);  view_27 = permute_14 = None
        view_28: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [1, 1000, 1024]);  mm_8 = None
        view_29: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_28, [1, 1000, -1, 256]);  view_28 = None
        permute_15: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_27: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_15, unsqueeze_21);  unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_10: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_15, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_10);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_9: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_15, 3, 0, 128);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_3: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_3, slice_9], -1);  neg_3 = slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_28: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_22);  cat_3 = unsqueeze_22 = None
        add_22: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, mul_28);  mul_27 = mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_23: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_22, 2)
        expand_9: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_23, [1, 4, 2, 1000, 256]);  unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_8: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_33: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1, 8, 1000, 256]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_30: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [1000, 2304]);  convert_element_type_35 = None
        permute_16: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_9: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_30, permute_16);  view_30 = permute_16 = None
        view_31: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [1, 1000, 1024]);  mm_9 = None
        view_32: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_31, [1, 1000, -1, 256]);  view_31 = None
        permute_17: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_24: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_17, 2)
        expand_10: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_24, [1, 4, 2, 1000, 256]);  unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_9: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_34: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1, 8, 1000, 256]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[1000][1]cuda:0" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[1000][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_4: "i64[1, 1000][1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_5: "i64[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 1, 1000][1000, 1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[1000][1]cuda:0" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1000][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_1: "i64[1, 1000][1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_2: "i64[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 1);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 1000, 1][1000, 1000, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_6, unsqueeze_3);  unsqueeze_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.expand.default(le, [1, -1, 1000, 1000]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        expand_11: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_1, [1, 8, 1000, 1000]);  where_1 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_21, view_33, view_34, expand_11, False, scale = 0.0625);  add_21 = view_33 = view_34 = expand_11 = None
        getitem_4: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_35: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_18, [1, 1000, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_36: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_35, [1000, 2048]);  view_35 = None
        permute_19: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_10: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_36, permute_19);  view_36 = permute_19 = None
        view_37: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [1, 1000, 2304]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_44: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_6: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_44, 2)
        mean_5: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
        add_23: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_29: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, rsqrt_5);  convert_element_type_44 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_45: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg20_1, torch.float32);  arg20_1 = None
        add_24: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_45, 1.0);  convert_element_type_45 = None
        mul_30: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, add_24);  mul_29 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_46: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_25: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_18, convert_element_type_46);  add_18 = convert_element_type_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_47: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_7: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_47, 2)
        mean_6: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_26: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_31: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_47, rsqrt_6);  convert_element_type_47 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_48: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None
        add_27: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_48, 1.0);  convert_element_type_48 = None
        mul_32: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, add_27);  mul_31 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_49: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_38: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [1000, 2304])
        permute_20: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_11: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_38, permute_20);  view_38 = permute_20 = None
        view_39: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [1, 1000, 9216]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_52: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        mul_37: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, 0.5)
        mul_33: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, convert_element_type_52)
        mul_34: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, convert_element_type_52);  mul_33 = None
        mul_35: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, 0.044715);  mul_34 = None
        add_28: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_52, mul_35);  convert_element_type_52 = mul_35 = None
        mul_36: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
        tanh_1: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_36);  mul_36 = None
        add_29: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1);  tanh_1 = None
        mul_38: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, add_29);  mul_37 = add_29 = None
        convert_element_type_53: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_40: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [1000, 2304]);  convert_element_type_49 = None
        permute_21: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_12: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_40, permute_21);  view_40 = permute_21 = None
        view_41: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [1, 1000, 9216]);  mm_12 = None
        mul_39: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, view_41);  convert_element_type_53 = view_41 = None
        view_42: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_39, [1000, 9216]);  mul_39 = None
        permute_22: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_13: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_42, permute_22);  view_42 = permute_22 = None
        view_43: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [1, 1000, 2304]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_58: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_8: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_58, 2)
        mean_7: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        add_30: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_40: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, rsqrt_7);  convert_element_type_58 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_59: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg25_1, torch.float32);  arg25_1 = None
        add_31: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_59, 1.0);  convert_element_type_59 = None
        mul_41: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, add_31);  mul_40 = add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_60: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_32: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_25, convert_element_type_60);  add_25 = convert_element_type_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_61: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_9: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_61, 2)
        mean_8: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        add_33: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_42: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, rsqrt_8);  convert_element_type_61 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_62: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg26_1, torch.float32);  arg26_1 = None
        add_34: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_62, 1.0);  convert_element_type_62 = None
        mul_43: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, add_34);  mul_42 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_63: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_44: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [1000, 2304])
        permute_23: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_14: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_44, permute_23);  view_44 = permute_23 = None
        view_45: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [1, 1000, 2048]);  mm_14 = None
        view_46: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [1, 1000, -1, 256]);  view_45 = None
        permute_24: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_25: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_44: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_24, unsqueeze_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_12: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_24, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_12);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_11: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_24, 3, 0, 128);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_4: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_4, slice_11], -1);  neg_4 = slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_26: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_45: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_26);  cat_4 = None
        add_35: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_47: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [1000, 2304])
        permute_25: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_15: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_47, permute_25);  view_47 = permute_25 = None
        view_48: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [1, 1000, 1024]);  mm_15 = None
        view_49: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [1, 1000, -1, 256]);  view_48 = None
        permute_26: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_46: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_26, unsqueeze_25);  unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_14: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_26, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_5: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_14);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_13: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_26, 3, 0, 128);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_5: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_5, slice_13], -1);  neg_5 = slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_47: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_26);  cat_5 = unsqueeze_26 = None
        add_36: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_27: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_36, 2)
        expand_12: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_27, [1, 4, 2, 1000, 256]);  unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_12: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_53: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [1, 8, 1000, 256]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_50: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [1000, 2304]);  convert_element_type_63 = None
        permute_27: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_16: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_50, permute_27);  view_50 = permute_27 = None
        view_51: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [1, 1000, 1024]);  mm_16 = None
        view_52: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_51, [1, 1000, -1, 256]);  view_51 = None
        permute_28: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_28: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_28, 2)
        expand_13: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_28, [1, 4, 2, 1000, 256]);  unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_13: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_54: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [1, 8, 1000, 256]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        expand_14: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_2, [1, 8, 1000, 1000]);  where_2 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_35, view_53, view_54, expand_14, False, scale = 0.0625);  add_35 = view_53 = view_54 = expand_14 = None
        getitem_8: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_55: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_29, [1, 1000, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_56: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_55, [1000, 2048]);  view_55 = None
        permute_30: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_17: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_56, permute_30);  view_56 = permute_30 = None
        view_57: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [1, 1000, 2304]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_73: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_10: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 2)
        mean_9: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_37: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_48: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_73, rsqrt_9);  convert_element_type_73 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_74: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg31_1, torch.float32);  arg31_1 = None
        add_38: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_74, 1.0);  convert_element_type_74 = None
        mul_49: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, add_38);  mul_48 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_75: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_39: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_32, convert_element_type_75);  add_32 = convert_element_type_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_76: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_11: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_76, 2)
        mean_10: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_40: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_50: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_76, rsqrt_10);  convert_element_type_76 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_77: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg32_1, torch.float32);  arg32_1 = None
        add_41: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_77, 1.0);  convert_element_type_77 = None
        mul_51: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_41);  mul_50 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_78: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_51, torch.bfloat16);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_58: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_78, [1000, 2304])
        permute_31: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_18: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_58, permute_31);  view_58 = permute_31 = None
        view_59: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [1, 1000, 9216]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_81: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        mul_56: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, 0.5)
        mul_52: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, convert_element_type_81)
        mul_53: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, convert_element_type_81);  mul_52 = None
        mul_54: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, 0.044715);  mul_53 = None
        add_42: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_81, mul_54);  convert_element_type_81 = mul_54 = None
        mul_55: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_42, 0.7978845608028654);  add_42 = None
        tanh_2: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_55);  mul_55 = None
        add_43: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1);  tanh_2 = None
        mul_57: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, add_43);  mul_56 = add_43 = None
        convert_element_type_82: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_60: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_78, [1000, 2304]);  convert_element_type_78 = None
        permute_32: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_19: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_60, permute_32);  view_60 = permute_32 = None
        view_61: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [1, 1000, 9216]);  mm_19 = None
        mul_58: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_82, view_61);  convert_element_type_82 = view_61 = None
        view_62: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_58, [1000, 9216]);  mul_58 = None
        permute_33: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_20: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_62, permute_33);  view_62 = permute_33 = None
        view_63: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [1, 1000, 2304]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_87: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_12: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2)
        mean_11: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
        add_44: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_59: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_87, rsqrt_11);  convert_element_type_87 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_88: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg36_1, torch.float32);  arg36_1 = None
        add_45: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_88, 1.0);  convert_element_type_88 = None
        mul_60: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, add_45);  mul_59 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_89: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_60, torch.bfloat16);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_46: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_39, convert_element_type_89);  add_39 = convert_element_type_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_90: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_13: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_90, 2)
        mean_12: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_47: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_61: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_90, rsqrt_12);  convert_element_type_90 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_91: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg37_1, torch.float32);  arg37_1 = None
        add_48: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_91, 1.0);  convert_element_type_91 = None
        mul_62: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, add_48);  mul_61 = add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_92: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_64: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [1000, 2304])
        permute_34: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_21: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_64, permute_34);  view_64 = permute_34 = None
        view_65: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [1, 1000, 2048]);  mm_21 = None
        view_66: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_65, [1, 1000, -1, 256]);  view_65 = None
        permute_35: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_29: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_63: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_35, unsqueeze_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_18: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_35, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_18);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_17: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_35, 3, 0, 128);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_6: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_6, slice_17], -1);  neg_6 = slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_30: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_64: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_30);  cat_6 = None
        add_49: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_67: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [1000, 2304])
        permute_36: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_22: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_67, permute_36);  view_67 = permute_36 = None
        view_68: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [1, 1000, 1024]);  mm_22 = None
        view_69: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_68, [1, 1000, -1, 256]);  view_68 = None
        permute_37: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_65: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_37, unsqueeze_29);  unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_20: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_37, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_20);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_19: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_37, 3, 0, 128);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_7: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_7, slice_19], -1);  neg_7 = slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_66: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_30);  cat_7 = unsqueeze_30 = None
        add_50: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_31: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_50, 2)
        expand_15: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_31, [1, 4, 2, 1000, 256]);  unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_16: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_73: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [1, 8, 1000, 256]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_70: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [1000, 2304]);  convert_element_type_92 = None
        permute_38: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_23: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_70, permute_38);  view_70 = permute_38 = None
        view_71: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [1, 1000, 1024]);  mm_23 = None
        view_72: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_71, [1, 1000, -1, 256]);  view_71 = None
        permute_39: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_32: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_39, 2)
        expand_16: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_32, [1, 4, 2, 1000, 256]);  unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_17: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_74: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1, 8, 1000, 256]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        expand_17: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_3, [1, 8, 1000, 1000]);  where_3 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_49, view_73, view_74, expand_17, False, scale = 0.0625);  add_49 = view_73 = view_74 = expand_17 = None
        getitem_12: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_75: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_40, [1, 1000, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_76: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_75, [1000, 2048]);  view_75 = None
        permute_41: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_24: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_41);  view_76 = permute_41 = None
        view_77: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [1, 1000, 2304]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_101: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_14: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_101, 2)
        mean_13: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        add_51: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_67: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_101, rsqrt_13);  convert_element_type_101 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_102: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg42_1, torch.float32);  arg42_1 = None
        add_52: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_102, 1.0);  convert_element_type_102 = None
        mul_68: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, add_52);  mul_67 = add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_103: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_68, torch.bfloat16);  mul_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_53: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_46, convert_element_type_103);  add_46 = convert_element_type_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_104: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_15: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_104, 2)
        mean_14: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        add_54: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_69: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_104, rsqrt_14);  convert_element_type_104 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_105: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg43_1, torch.float32);  arg43_1 = None
        add_55: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_105, 1.0);  convert_element_type_105 = None
        mul_70: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_69, add_55);  mul_69 = add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_106: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_78: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_106, [1000, 2304])
        permute_42: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_25: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_78, permute_42);  view_78 = permute_42 = None
        view_79: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [1, 1000, 9216]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_109: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        mul_75: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_109, 0.5)
        mul_71: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_109, convert_element_type_109)
        mul_72: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, convert_element_type_109);  mul_71 = None
        mul_73: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, 0.044715);  mul_72 = None
        add_56: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_109, mul_73);  convert_element_type_109 = mul_73 = None
        mul_74: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_56, 0.7978845608028654);  add_56 = None
        tanh_3: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_74);  mul_74 = None
        add_57: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1);  tanh_3 = None
        mul_76: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, add_57);  mul_75 = add_57 = None
        convert_element_type_110: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_80: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_106, [1000, 2304]);  convert_element_type_106 = None
        permute_43: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_26: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_80, permute_43);  view_80 = permute_43 = None
        view_81: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [1, 1000, 9216]);  mm_26 = None
        mul_77: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_110, view_81);  convert_element_type_110 = view_81 = None
        view_82: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_77, [1000, 9216]);  mul_77 = None
        permute_44: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_27: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_82, permute_44);  view_82 = permute_44 = None
        view_83: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [1, 1000, 2304]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_115: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_83, torch.float32);  view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_16: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_115, 2)
        mean_15: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_58: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_78: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, rsqrt_15);  convert_element_type_115 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_116: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg47_1, torch.float32);  arg47_1 = None
        add_59: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_116, 1.0);  convert_element_type_116 = None
        mul_79: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, add_59);  mul_78 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_117: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_60: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_53, convert_element_type_117);  add_53 = convert_element_type_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_118: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_60, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_17: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_118, 2)
        mean_16: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_61: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_80: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_118, rsqrt_16);  convert_element_type_118 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_119: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg48_1, torch.float32);  arg48_1 = None
        add_62: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_119, 1.0);  convert_element_type_119 = None
        mul_81: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, add_62);  mul_80 = add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_120: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_84: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_120, [1000, 2304])
        permute_45: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_28: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_84, permute_45);  view_84 = permute_45 = None
        view_85: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [1, 1000, 2048]);  mm_28 = None
        view_86: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_85, [1, 1000, -1, 256]);  view_85 = None
        permute_46: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_33: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_82: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_46, unsqueeze_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_22: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_46, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_8: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_22);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_21: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_46, 3, 0, 128);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_8: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_8, slice_21], -1);  neg_8 = slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_34: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_83: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_34);  cat_8 = None
        add_63: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_87: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_120, [1000, 2304])
        permute_47: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_29: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_87, permute_47);  view_87 = permute_47 = None
        view_88: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [1, 1000, 1024]);  mm_29 = None
        view_89: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_88, [1, 1000, -1, 256]);  view_88 = None
        permute_48: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_84: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_48, unsqueeze_33);  unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_24: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_48, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_24);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_23: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_48, 3, 0, 128);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_9: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_9, slice_23], -1);  neg_9 = slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_85: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_34);  cat_9 = unsqueeze_34 = None
        add_64: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_35: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_64, 2)
        expand_18: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_35, [1, 4, 2, 1000, 256]);  unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_20: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_93: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [1, 8, 1000, 256]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_90: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_120, [1000, 2304]);  convert_element_type_120 = None
        permute_49: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_30: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_90, permute_49);  view_90 = permute_49 = None
        view_91: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [1, 1000, 1024]);  mm_30 = None
        view_92: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_91, [1, 1000, -1, 256]);  view_91 = None
        permute_50: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_36: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_50, 2)
        expand_19: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_36, [1, 4, 2, 1000, 256]);  unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_21: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_94: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [1, 8, 1000, 256]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        expand_20: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_4, [1, 8, 1000, 1000]);  where_4 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_63, view_93, view_94, expand_20, False, scale = 0.0625);  add_63 = view_93 = view_94 = expand_20 = None
        getitem_16: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_95: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [1, 1000, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_96: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_95, [1000, 2048]);  view_95 = None
        permute_52: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_31: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_96, permute_52);  view_96 = permute_52 = None
        view_97: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [1, 1000, 2304]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_130: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_18: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_130, 2)
        mean_17: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
        add_65: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_86: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_130, rsqrt_17);  convert_element_type_130 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_131: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg53_1, torch.float32);  arg53_1 = None
        add_66: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_131, 1.0);  convert_element_type_131 = None
        mul_87: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, add_66);  mul_86 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_132: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_67: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_60, convert_element_type_132);  add_60 = convert_element_type_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_133: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_19: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_133, 2)
        mean_18: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_68: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_88: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_133, rsqrt_18);  convert_element_type_133 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_134: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg54_1, torch.float32);  arg54_1 = None
        add_69: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_134, 1.0);  convert_element_type_134 = None
        mul_89: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, add_69);  mul_88 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_135: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_98: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [1000, 2304])
        permute_53: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_32: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_98, permute_53);  view_98 = permute_53 = None
        view_99: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [1, 1000, 9216]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_138: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        mul_94: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_138, 0.5)
        mul_90: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_138, convert_element_type_138)
        mul_91: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, convert_element_type_138);  mul_90 = None
        mul_92: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, 0.044715);  mul_91 = None
        add_70: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_138, mul_92);  convert_element_type_138 = mul_92 = None
        mul_93: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_4: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_93);  mul_93 = None
        add_71: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1);  tanh_4 = None
        mul_95: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, add_71);  mul_94 = add_71 = None
        convert_element_type_139: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_95, torch.bfloat16);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_100: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [1000, 2304]);  convert_element_type_135 = None
        permute_54: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_33: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_100, permute_54);  view_100 = permute_54 = None
        view_101: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [1, 1000, 9216]);  mm_33 = None
        mul_96: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, view_101);  convert_element_type_139 = view_101 = None
        view_102: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_96, [1000, 9216]);  mul_96 = None
        permute_55: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        mm_34: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_102, permute_55);  view_102 = permute_55 = None
        view_103: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [1, 1000, 2304]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_144: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_103, torch.float32);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_20: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_144, 2)
        mean_19: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        add_72: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_97: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_144, rsqrt_19);  convert_element_type_144 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_145: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg58_1, torch.float32);  arg58_1 = None
        add_73: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_145, 1.0);  convert_element_type_145 = None
        mul_98: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, add_73);  mul_97 = add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_146: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_74: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_67, convert_element_type_146);  add_67 = convert_element_type_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_147: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_21: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_147, 2)
        mean_20: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        add_75: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_99: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_147, rsqrt_20);  convert_element_type_147 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_148: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg59_1, torch.float32);  arg59_1 = None
        add_76: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_148, 1.0);  convert_element_type_148 = None
        mul_100: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, add_76);  mul_99 = add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_149: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_100, torch.bfloat16);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_104: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_149, [1000, 2304])
        permute_56: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_35: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_104, permute_56);  view_104 = permute_56 = None
        view_105: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [1, 1000, 2048]);  mm_35 = None
        view_106: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [1, 1000, -1, 256]);  view_105 = None
        permute_57: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_37: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_101: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_57, unsqueeze_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_28: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_57, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_28);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_27: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_57, 3, 0, 128);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_10: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_10, slice_27], -1);  neg_10 = slice_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_38: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_102: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_38);  cat_10 = None
        add_77: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_107: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_149, [1000, 2304])
        permute_58: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_36: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_107, permute_58);  view_107 = permute_58 = None
        view_108: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [1, 1000, 1024]);  mm_36 = None
        view_109: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_108, [1, 1000, -1, 256]);  view_108 = None
        permute_59: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_103: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_59, unsqueeze_37);  unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_30: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_59, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_11: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_30);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_29: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_59, 3, 0, 128);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_11: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_11, slice_29], -1);  neg_11 = slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_104: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_38);  cat_11 = unsqueeze_38 = None
        add_78: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_103, mul_104);  mul_103 = mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_39: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_78, 2)
        expand_21: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_39, [1, 4, 2, 1000, 256]);  unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_24: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_113: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [1, 8, 1000, 256]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_110: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_149, [1000, 2304]);  convert_element_type_149 = None
        permute_60: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        mm_37: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_110, permute_60);  view_110 = permute_60 = None
        view_111: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [1, 1000, 1024]);  mm_37 = None
        view_112: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [1, 1000, -1, 256]);  view_111 = None
        permute_61: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_40: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_61, 2)
        expand_22: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_40, [1, 4, 2, 1000, 256]);  unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_25: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_114: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [1, 8, 1000, 256]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        expand_23: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_5, [1, 8, 1000, 1000]);  where_5 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_77, view_113, view_114, expand_23, False, scale = 0.0625);  add_77 = view_113 = view_114 = expand_23 = None
        getitem_20: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_115: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_62, [1, 1000, -1]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_116: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_115, [1000, 2048]);  view_115 = None
        permute_63: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_38: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_116, permute_63);  view_116 = permute_63 = None
        view_117: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [1, 1000, 2304]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_158: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_22: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_158, 2)
        mean_21: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        add_79: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_105: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_158, rsqrt_21);  convert_element_type_158 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_159: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg64_1, torch.float32);  arg64_1 = None
        add_80: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_159, 1.0);  convert_element_type_159 = None
        mul_106: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, add_80);  mul_105 = add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_160: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_106, torch.bfloat16);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_81: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_74, convert_element_type_160);  add_74 = convert_element_type_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_161: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_23: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_161, 2)
        mean_22: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        add_82: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_107: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, rsqrt_22);  convert_element_type_161 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_162: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg65_1, torch.float32);  arg65_1 = None
        add_83: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_162, 1.0);  convert_element_type_162 = None
        mul_108: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, add_83);  mul_107 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_163: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_118: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_163, [1000, 2304])
        permute_64: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_39: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_118, permute_64);  view_118 = permute_64 = None
        view_119: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [1, 1000, 9216]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_166: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        mul_113: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_166, 0.5)
        mul_109: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_166, convert_element_type_166)
        mul_110: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, convert_element_type_166);  mul_109 = None
        mul_111: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, 0.044715);  mul_110 = None
        add_84: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_166, mul_111);  convert_element_type_166 = mul_111 = None
        mul_112: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_84, 0.7978845608028654);  add_84 = None
        tanh_5: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_112);  mul_112 = None
        add_85: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1);  tanh_5 = None
        mul_114: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, add_85);  mul_113 = add_85 = None
        convert_element_type_167: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_114, torch.bfloat16);  mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_120: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_163, [1000, 2304]);  convert_element_type_163 = None
        permute_65: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_40: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_120, permute_65);  view_120 = permute_65 = None
        view_121: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [1, 1000, 9216]);  mm_40 = None
        mul_115: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_167, view_121);  convert_element_type_167 = view_121 = None
        view_122: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_115, [1000, 9216]);  mul_115 = None
        permute_66: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_41: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_122, permute_66);  view_122 = permute_66 = None
        view_123: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [1, 1000, 2304]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_172: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_123, torch.float32);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_24: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_172, 2)
        mean_23: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
        add_86: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_116: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_172, rsqrt_23);  convert_element_type_172 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_173: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg69_1, torch.float32);  arg69_1 = None
        add_87: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_173, 1.0);  convert_element_type_173 = None
        mul_117: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, add_87);  mul_116 = add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_174: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_88: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_81, convert_element_type_174);  add_81 = convert_element_type_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_175: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_25: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_175, 2)
        mean_24: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_89: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_118: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, rsqrt_24);  convert_element_type_175 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_176: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg70_1, torch.float32);  arg70_1 = None
        add_90: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_176, 1.0);  convert_element_type_176 = None
        mul_119: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, add_90);  mul_118 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_177: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_124: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_177, [1000, 2304])
        permute_67: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_42: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_124, permute_67);  view_124 = permute_67 = None
        view_125: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [1, 1000, 2048]);  mm_42 = None
        view_126: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [1, 1000, -1, 256]);  view_125 = None
        permute_68: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_41: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_120: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_68, unsqueeze_41)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_32: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_68, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_12: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_32);  slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_31: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_68, 3, 0, 128);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_12: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_12, slice_31], -1);  neg_12 = slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_42: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_121: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_42);  cat_12 = None
        add_91: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_127: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_177, [1000, 2304])
        permute_69: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_43: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_127, permute_69);  view_127 = permute_69 = None
        view_128: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [1, 1000, 1024]);  mm_43 = None
        view_129: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_128, [1, 1000, -1, 256]);  view_128 = None
        permute_70: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_122: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_70, unsqueeze_41);  unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_34: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_70, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_13: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_34);  slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_33: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_70, 3, 0, 128);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_13: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_13, slice_33], -1);  neg_13 = slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_123: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_42);  cat_13 = unsqueeze_42 = None
        add_92: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_43: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_92, 2)
        expand_24: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_43, [1, 4, 2, 1000, 256]);  unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_28: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_133: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [1, 8, 1000, 256]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_130: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_177, [1000, 2304]);  convert_element_type_177 = None
        permute_71: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_44: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_130, permute_71);  view_130 = permute_71 = None
        view_131: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [1, 1000, 1024]);  mm_44 = None
        view_132: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_131, [1, 1000, -1, 256]);  view_131 = None
        permute_72: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_44: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_72, 2)
        expand_25: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_44, [1, 4, 2, 1000, 256]);  unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_29: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_134: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [1, 8, 1000, 256]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        expand_26: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_6, [1, 8, 1000, 1000]);  where_6 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_91, view_133, view_134, expand_26, False, scale = 0.0625);  add_91 = view_133 = view_134 = expand_26 = None
        getitem_24: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_24, [0, 2, 1, 3]);  getitem_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_135: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_73, [1, 1000, -1]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_136: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_135, [1000, 2048]);  view_135 = None
        permute_74: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_45: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_136, permute_74);  view_136 = permute_74 = None
        view_137: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [1, 1000, 2304]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_187: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_26: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_187, 2)
        mean_25: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        add_93: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_124: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_187, rsqrt_25);  convert_element_type_187 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_188: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg75_1, torch.float32);  arg75_1 = None
        add_94: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_188, 1.0);  convert_element_type_188 = None
        mul_125: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, add_94);  mul_124 = add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_189: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_95: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_88, convert_element_type_189);  add_88 = convert_element_type_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_190: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_27: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_190, 2)
        mean_26: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_96: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_126: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_190, rsqrt_26);  convert_element_type_190 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_191: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg76_1, torch.float32);  arg76_1 = None
        add_97: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_191, 1.0);  convert_element_type_191 = None
        mul_127: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, add_97);  mul_126 = add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_192: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_127, torch.bfloat16);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_138: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_192, [1000, 2304])
        permute_75: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_46: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_138, permute_75);  view_138 = permute_75 = None
        view_139: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [1, 1000, 9216]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_195: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        mul_132: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_195, 0.5)
        mul_128: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_195, convert_element_type_195)
        mul_129: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, convert_element_type_195);  mul_128 = None
        mul_130: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 0.044715);  mul_129 = None
        add_98: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_195, mul_130);  convert_element_type_195 = mul_130 = None
        mul_131: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
        tanh_6: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_131);  mul_131 = None
        add_99: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1);  tanh_6 = None
        mul_133: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, add_99);  mul_132 = add_99 = None
        convert_element_type_196: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_140: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_192, [1000, 2304]);  convert_element_type_192 = None
        permute_76: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_47: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_140, permute_76);  view_140 = permute_76 = None
        view_141: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [1, 1000, 9216]);  mm_47 = None
        mul_134: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_196, view_141);  convert_element_type_196 = view_141 = None
        view_142: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_134, [1000, 9216]);  mul_134 = None
        permute_77: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_48: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_142, permute_77);  view_142 = permute_77 = None
        view_143: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [1, 1000, 2304]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_201: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_28: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_201, 2)
        mean_27: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        add_100: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_135: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_201, rsqrt_27);  convert_element_type_201 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_202: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg80_1, torch.float32);  arg80_1 = None
        add_101: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_202, 1.0);  convert_element_type_202 = None
        mul_136: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, add_101);  mul_135 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_203: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_102: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_95, convert_element_type_203);  add_95 = convert_element_type_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_204: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_29: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_204, 2)
        mean_28: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
        add_103: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_137: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_204, rsqrt_28);  convert_element_type_204 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_205: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg81_1, torch.float32);  arg81_1 = None
        add_104: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_205, 1.0);  convert_element_type_205 = None
        mul_138: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, add_104);  mul_137 = add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_206: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_138, torch.bfloat16);  mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_144: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_206, [1000, 2304])
        permute_78: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_49: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_144, permute_78);  view_144 = permute_78 = None
        view_145: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [1, 1000, 2048]);  mm_49 = None
        view_146: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_145, [1, 1000, -1, 256]);  view_145 = None
        permute_79: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_45: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_139: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_79, unsqueeze_45)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_38: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_79, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_14: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_38);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_37: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_79, 3, 0, 128);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_14: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_14, slice_37], -1);  neg_14 = slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_46: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_140: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_14, unsqueeze_46);  cat_14 = None
        add_105: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_147: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_206, [1000, 2304])
        permute_80: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_50: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_147, permute_80);  view_147 = permute_80 = None
        view_148: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [1, 1000, 1024]);  mm_50 = None
        view_149: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_148, [1, 1000, -1, 256]);  view_148 = None
        permute_81: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_141: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_81, unsqueeze_45);  unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_40: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_81, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_15: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_40);  slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_39: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_81, 3, 0, 128);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_15: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_15, slice_39], -1);  neg_15 = slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_142: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_15, unsqueeze_46);  cat_15 = unsqueeze_46 = None
        add_106: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_47: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_106, 2)
        expand_27: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_47, [1, 4, 2, 1000, 256]);  unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_32: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_153: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [1, 8, 1000, 256]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_150: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_206, [1000, 2304]);  convert_element_type_206 = None
        permute_82: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        mm_51: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_150, permute_82);  view_150 = permute_82 = None
        view_151: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [1, 1000, 1024]);  mm_51 = None
        view_152: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_151, [1, 1000, -1, 256]);  view_151 = None
        permute_83: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_48: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_83, 2)
        expand_28: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_48, [1, 4, 2, 1000, 256]);  unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_33: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_154: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [1, 8, 1000, 256]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_19: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        expand_29: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_7, [1, 8, 1000, 1000]);  where_7 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_105, view_153, view_154, expand_29, False, scale = 0.0625);  add_105 = view_153 = view_154 = expand_29 = None
        getitem_28: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_155: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_84, [1, 1000, -1]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_156: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [1000, 2048]);  view_155 = None
        permute_85: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_52: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_156, permute_85);  view_156 = permute_85 = None
        view_157: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [1, 1000, 2304]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_215: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_30: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_215, 2)
        mean_29: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        add_107: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_143: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_215, rsqrt_29);  convert_element_type_215 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_216: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg86_1, torch.float32);  arg86_1 = None
        add_108: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_216, 1.0);  convert_element_type_216 = None
        mul_144: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, add_108);  mul_143 = add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_217: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_109: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_102, convert_element_type_217);  add_102 = convert_element_type_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_218: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_31: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_218, 2)
        mean_30: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_110: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_145: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_218, rsqrt_30);  convert_element_type_218 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_219: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg87_1, torch.float32);  arg87_1 = None
        add_111: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_219, 1.0);  convert_element_type_219 = None
        mul_146: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, add_111);  mul_145 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_220: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_146, torch.bfloat16);  mul_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_158: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_220, [1000, 2304])
        permute_86: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        mm_53: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_158, permute_86);  view_158 = permute_86 = None
        view_159: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [1, 1000, 9216]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_223: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        mul_151: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_223, 0.5)
        mul_147: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_223, convert_element_type_223)
        mul_148: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, convert_element_type_223);  mul_147 = None
        mul_149: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, 0.044715);  mul_148 = None
        add_112: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_223, mul_149);  convert_element_type_223 = mul_149 = None
        mul_150: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_112, 0.7978845608028654);  add_112 = None
        tanh_7: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_113: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1);  tanh_7 = None
        mul_152: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, add_113);  mul_151 = add_113 = None
        convert_element_type_224: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_160: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_220, [1000, 2304]);  convert_element_type_220 = None
        permute_87: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_54: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_160, permute_87);  view_160 = permute_87 = None
        view_161: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [1, 1000, 9216]);  mm_54 = None
        mul_153: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_224, view_161);  convert_element_type_224 = view_161 = None
        view_162: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_153, [1000, 9216]);  mul_153 = None
        permute_88: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_55: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_162, permute_88);  view_162 = permute_88 = None
        view_163: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [1, 1000, 2304]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_229: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_163, torch.float32);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_32: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_229, 2)
        mean_31: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        add_114: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_154: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_229, rsqrt_31);  convert_element_type_229 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_230: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg91_1, torch.float32);  arg91_1 = None
        add_115: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_230, 1.0);  convert_element_type_230 = None
        mul_155: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, add_115);  mul_154 = add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_231: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_155, torch.bfloat16);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_116: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, convert_element_type_231);  add_109 = convert_element_type_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_232: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_116, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_33: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_232, 2)
        mean_32: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None
        add_117: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        mul_156: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_232, rsqrt_32);  convert_element_type_232 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_233: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg92_1, torch.float32);  arg92_1 = None
        add_118: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_233, 1.0);  convert_element_type_233 = None
        mul_157: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, add_118);  mul_156 = add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_234: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_164: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_234, [1000, 2304])
        permute_89: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        mm_56: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_164, permute_89);  view_164 = permute_89 = None
        view_165: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [1, 1000, 2048]);  mm_56 = None
        view_166: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_165, [1, 1000, -1, 256]);  view_165 = None
        permute_90: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_49: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_158: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_90, unsqueeze_49)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_42: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_90, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_16: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_42);  slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_41: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_90, 3, 0, 128);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_16: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_16, slice_41], -1);  neg_16 = slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_50: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_159: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_16, unsqueeze_50);  cat_16 = None
        add_119: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_167: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_234, [1000, 2304])
        permute_91: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_57: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_167, permute_91);  view_167 = permute_91 = None
        view_168: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [1, 1000, 1024]);  mm_57 = None
        view_169: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_168, [1, 1000, -1, 256]);  view_168 = None
        permute_92: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_160: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_92, unsqueeze_49);  unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_44: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_92, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_17: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_44);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_43: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_92, 3, 0, 128);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_17: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_17, slice_43], -1);  neg_17 = slice_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_161: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_17, unsqueeze_50);  cat_17 = unsqueeze_50 = None
        add_120: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_51: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_120, 2)
        expand_30: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_51, [1, 4, 2, 1000, 256]);  unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_36: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_173: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [1, 8, 1000, 256]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_170: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_234, [1000, 2304]);  convert_element_type_234 = None
        permute_93: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_58: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_170, permute_93);  view_170 = permute_93 = None
        view_171: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [1, 1000, 1024]);  mm_58 = None
        view_172: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_171, [1, 1000, -1, 256]);  view_171 = None
        permute_94: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_52: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_94, 2)
        expand_31: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_52, [1, 4, 2, 1000, 256]);  unsqueeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_37: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_174: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [1, 8, 1000, 256]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_22: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        expand_32: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_8, [1, 8, 1000, 1000]);  where_8 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_119, view_173, view_174, expand_32, False, scale = 0.0625);  add_119 = view_173 = view_174 = expand_32 = None
        getitem_32: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_32, [0, 2, 1, 3]);  getitem_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_175: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_95, [1, 1000, -1]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_176: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_175, [1000, 2048]);  view_175 = None
        permute_96: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_59: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_96);  view_176 = permute_96 = None
        view_177: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [1, 1000, 2304]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_244: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_34: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_244, 2)
        mean_33: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
        add_121: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        mul_162: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_244, rsqrt_33);  convert_element_type_244 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_245: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg97_1, torch.float32);  arg97_1 = None
        add_122: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_245, 1.0);  convert_element_type_245 = None
        mul_163: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, add_122);  mul_162 = add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_246: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_163, torch.bfloat16);  mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_123: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_116, convert_element_type_246);  add_116 = convert_element_type_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_247: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_123, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_35: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_247, 2)
        mean_34: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        add_124: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_164: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_247, rsqrt_34);  convert_element_type_247 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_248: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg98_1, torch.float32);  arg98_1 = None
        add_125: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_248, 1.0);  convert_element_type_248 = None
        mul_165: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, add_125);  mul_164 = add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_249: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_178: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_249, [1000, 2304])
        permute_97: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_60: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_178, permute_97);  view_178 = permute_97 = None
        view_179: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [1, 1000, 9216]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_252: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        mul_170: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_252, 0.5)
        mul_166: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_252, convert_element_type_252)
        mul_167: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, convert_element_type_252);  mul_166 = None
        mul_168: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, 0.044715);  mul_167 = None
        add_126: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_252, mul_168);  convert_element_type_252 = mul_168 = None
        mul_169: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, 0.7978845608028654);  add_126 = None
        tanh_8: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_169);  mul_169 = None
        add_127: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1);  tanh_8 = None
        mul_171: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, add_127);  mul_170 = add_127 = None
        convert_element_type_253: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_171, torch.bfloat16);  mul_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_180: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_249, [1000, 2304]);  convert_element_type_249 = None
        permute_98: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_61: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_180, permute_98);  view_180 = permute_98 = None
        view_181: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [1, 1000, 9216]);  mm_61 = None
        mul_172: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_253, view_181);  convert_element_type_253 = view_181 = None
        view_182: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_172, [1000, 9216]);  mul_172 = None
        permute_99: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_62: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_182, permute_99);  view_182 = permute_99 = None
        view_183: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [1, 1000, 2304]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_258: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_183, torch.float32);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_36: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_258, 2)
        mean_35: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
        add_128: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_173: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_258, rsqrt_35);  convert_element_type_258 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_259: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg102_1, torch.float32);  arg102_1 = None
        add_129: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_259, 1.0);  convert_element_type_259 = None
        mul_174: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, add_129);  mul_173 = add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_260: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_130: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, convert_element_type_260);  add_123 = convert_element_type_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_261: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_130, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_37: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_261, 2)
        mean_36: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None
        add_131: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_175: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_261, rsqrt_36);  convert_element_type_261 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_262: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg103_1, torch.float32);  arg103_1 = None
        add_132: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_262, 1.0);  convert_element_type_262 = None
        mul_176: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, add_132);  mul_175 = add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_263: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_176, torch.bfloat16);  mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_184: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_263, [1000, 2304])
        permute_100: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_63: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_184, permute_100);  view_184 = permute_100 = None
        view_185: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [1, 1000, 2048]);  mm_63 = None
        view_186: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [1, 1000, -1, 256]);  view_185 = None
        permute_101: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_53: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_177: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_101, unsqueeze_53)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_48: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_101, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_18: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_48);  slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_47: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_101, 3, 0, 128);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_18: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_18, slice_47], -1);  neg_18 = slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_54: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_178: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_18, unsqueeze_54);  cat_18 = None
        add_133: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_187: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_263, [1000, 2304])
        permute_102: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_64: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_187, permute_102);  view_187 = permute_102 = None
        view_188: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [1, 1000, 1024]);  mm_64 = None
        view_189: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_188, [1, 1000, -1, 256]);  view_188 = None
        permute_103: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_179: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_103, unsqueeze_53);  unsqueeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_50: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_103, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_19: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_50);  slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_49: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_103, 3, 0, 128);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_19: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_19, slice_49], -1);  neg_19 = slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_180: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_19, unsqueeze_54);  cat_19 = unsqueeze_54 = None
        add_134: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_55: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_134, 2)
        expand_33: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_55, [1, 4, 2, 1000, 256]);  unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_40: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_193: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [1, 8, 1000, 256]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_190: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_263, [1000, 2304]);  convert_element_type_263 = None
        permute_104: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        mm_65: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_190, permute_104);  view_190 = permute_104 = None
        view_191: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [1, 1000, 1024]);  mm_65 = None
        view_192: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [1, 1000, -1, 256]);  view_191 = None
        permute_105: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_56: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_105, 2)
        expand_34: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_56, [1, 4, 2, 1000, 256]);  unsqueeze_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_41: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_194: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [1, 8, 1000, 256]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_24: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_24, full_default_23);  full_default_24 = full_default_23 = None
        expand_35: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_9, [1, 8, 1000, 1000]);  where_9 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_133, view_193, view_194, expand_35, False, scale = 0.0625);  add_133 = view_193 = view_194 = expand_35 = None
        getitem_36: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_195: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_106, [1, 1000, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_196: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_195, [1000, 2048]);  view_195 = None
        permute_107: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_66: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_196, permute_107);  view_196 = permute_107 = None
        view_197: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [1, 1000, 2304]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_272: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_38: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_272, 2)
        mean_37: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
        add_135: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_181: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_272, rsqrt_37);  convert_element_type_272 = rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_273: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg108_1, torch.float32);  arg108_1 = None
        add_136: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_273, 1.0);  convert_element_type_273 = None
        mul_182: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, add_136);  mul_181 = add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_274: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_137: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, convert_element_type_274);  add_130 = convert_element_type_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_275: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_39: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_275, 2)
        mean_38: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        add_138: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_183: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_275, rsqrt_38);  convert_element_type_275 = rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_276: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg109_1, torch.float32);  arg109_1 = None
        add_139: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_276, 1.0);  convert_element_type_276 = None
        mul_184: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, add_139);  mul_183 = add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_277: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_184, torch.bfloat16);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_198: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_277, [1000, 2304])
        permute_108: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_67: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_108);  view_198 = permute_108 = None
        view_199: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [1, 1000, 9216]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_280: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        mul_189: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, 0.5)
        mul_185: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, convert_element_type_280)
        mul_186: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, convert_element_type_280);  mul_185 = None
        mul_187: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, 0.044715);  mul_186 = None
        add_140: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_280, mul_187);  convert_element_type_280 = mul_187 = None
        mul_188: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_140, 0.7978845608028654);  add_140 = None
        tanh_9: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
        add_141: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1);  tanh_9 = None
        mul_190: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, add_141);  mul_189 = add_141 = None
        convert_element_type_281: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16);  mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_200: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_277, [1000, 2304]);  convert_element_type_277 = None
        permute_109: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        mm_68: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_200, permute_109);  view_200 = permute_109 = None
        view_201: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [1, 1000, 9216]);  mm_68 = None
        mul_191: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_281, view_201);  convert_element_type_281 = view_201 = None
        view_202: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_191, [1000, 9216]);  mul_191 = None
        permute_110: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_69: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_202, permute_110);  view_202 = permute_110 = None
        view_203: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [1, 1000, 2304]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_286: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_203, torch.float32);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_40: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_286, 2)
        mean_39: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
        add_142: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_192: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_286, rsqrt_39);  convert_element_type_286 = rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_287: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg113_1, torch.float32);  arg113_1 = None
        add_143: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_287, 1.0);  convert_element_type_287 = None
        mul_193: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, add_143);  mul_192 = add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_288: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_144: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_137, convert_element_type_288);  add_137 = convert_element_type_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_289: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_41: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_289, 2)
        mean_40: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None
        add_145: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        mul_194: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_289, rsqrt_40);  convert_element_type_289 = rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_290: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg114_1, torch.float32);  arg114_1 = None
        add_146: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_290, 1.0);  convert_element_type_290 = None
        mul_195: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_194, add_146);  mul_194 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_291: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_195, torch.bfloat16);  mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_204: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_291, [1000, 2304])
        permute_111: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_70: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_204, permute_111);  view_204 = permute_111 = None
        view_205: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [1, 1000, 2048]);  mm_70 = None
        view_206: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_205, [1, 1000, -1, 256]);  view_205 = None
        permute_112: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_57: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_196: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_112, unsqueeze_57)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_52: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_112, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_20: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_52);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_51: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_112, 3, 0, 128);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_20: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_20, slice_51], -1);  neg_20 = slice_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_58: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_197: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_20, unsqueeze_58);  cat_20 = None
        add_147: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_207: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_291, [1000, 2304])
        permute_113: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        mm_71: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_207, permute_113);  view_207 = permute_113 = None
        view_208: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [1, 1000, 1024]);  mm_71 = None
        view_209: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_208, [1, 1000, -1, 256]);  view_208 = None
        permute_114: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_198: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_114, unsqueeze_57);  unsqueeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_54: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_114, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_21: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_54);  slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_53: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_114, 3, 0, 128);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_21: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_21, slice_53], -1);  neg_21 = slice_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_199: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_21, unsqueeze_58);  cat_21 = unsqueeze_58 = None
        add_148: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_59: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_148, 2)
        expand_36: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_59, [1, 4, 2, 1000, 256]);  unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_44: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_213: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [1, 8, 1000, 256]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_210: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_291, [1000, 2304]);  convert_element_type_291 = None
        permute_115: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_72: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_115);  view_210 = permute_115 = None
        view_211: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [1, 1000, 1024]);  mm_72 = None
        view_212: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_211, [1, 1000, -1, 256]);  view_211 = None
        permute_116: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_60: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_116, 2)
        expand_37: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_60, [1, 4, 2, 1000, 256]);  unsqueeze_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_45: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_214: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [1, 8, 1000, 256]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_27: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_27, full_default_26);  full_default_27 = full_default_26 = None
        expand_38: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_10, [1, 8, 1000, 1000]);  where_10 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_147, view_213, view_214, expand_38, False, scale = 0.0625);  add_147 = view_213 = view_214 = expand_38 = None
        getitem_40: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_40, [0, 2, 1, 3]);  getitem_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_117, [1, 1000, -1]);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_216: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_215, [1000, 2048]);  view_215 = None
        permute_118: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_73: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_216, permute_118);  view_216 = permute_118 = None
        view_217: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [1, 1000, 2304]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_301: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_42: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_301, 2)
        mean_41: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
        add_149: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_200: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_301, rsqrt_41);  convert_element_type_301 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_302: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg119_1, torch.float32);  arg119_1 = None
        add_150: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_302, 1.0);  convert_element_type_302 = None
        mul_201: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, add_150);  mul_200 = add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_303: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_151: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_144, convert_element_type_303);  add_144 = convert_element_type_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_304: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_43: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_304, 2)
        mean_42: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        add_152: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_42, 1e-06);  mean_42 = None
        rsqrt_42: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_202: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_304, rsqrt_42);  convert_element_type_304 = rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_305: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg120_1, torch.float32);  arg120_1 = None
        add_153: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_305, 1.0);  convert_element_type_305 = None
        mul_203: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_202, add_153);  mul_202 = add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_306: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_203, torch.bfloat16);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_218: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_306, [1000, 2304])
        permute_119: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_74: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_218, permute_119);  view_218 = permute_119 = None
        view_219: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [1, 1000, 9216]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_309: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        mul_208: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_309, 0.5)
        mul_204: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_309, convert_element_type_309)
        mul_205: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, convert_element_type_309);  mul_204 = None
        mul_206: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, 0.044715);  mul_205 = None
        add_154: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_309, mul_206);  convert_element_type_309 = mul_206 = None
        mul_207: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_154, 0.7978845608028654);  add_154 = None
        tanh_10: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_207);  mul_207 = None
        add_155: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1);  tanh_10 = None
        mul_209: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_208, add_155);  mul_208 = add_155 = None
        convert_element_type_310: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_209, torch.bfloat16);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_220: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_306, [1000, 2304]);  convert_element_type_306 = None
        permute_120: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_75: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_220, permute_120);  view_220 = permute_120 = None
        view_221: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [1, 1000, 9216]);  mm_75 = None
        mul_210: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_310, view_221);  convert_element_type_310 = view_221 = None
        view_222: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_210, [1000, 9216]);  mul_210 = None
        permute_121: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_76: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_222, permute_121);  view_222 = permute_121 = None
        view_223: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [1, 1000, 2304]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_315: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_44: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_315, 2)
        mean_43: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
        add_156: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_43, 1e-06);  mean_43 = None
        rsqrt_43: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        mul_211: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_315, rsqrt_43);  convert_element_type_315 = rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_316: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg124_1, torch.float32);  arg124_1 = None
        add_157: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_316, 1.0);  convert_element_type_316 = None
        mul_212: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, add_157);  mul_211 = add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_317: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_212, torch.bfloat16);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_158: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_151, convert_element_type_317);  add_151 = convert_element_type_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_318: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_45: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_318, 2)
        mean_44: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None
        add_159: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_44, 1e-06);  mean_44 = None
        rsqrt_44: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_213: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_318, rsqrt_44);  convert_element_type_318 = rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_319: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg125_1, torch.float32);  arg125_1 = None
        add_160: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_319, 1.0);  convert_element_type_319 = None
        mul_214: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, add_160);  mul_213 = add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_320: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_214, torch.bfloat16);  mul_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_224: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_320, [1000, 2304])
        permute_122: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_77: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_224, permute_122);  view_224 = permute_122 = None
        view_225: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [1, 1000, 2048]);  mm_77 = None
        view_226: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [1, 1000, -1, 256]);  view_225 = None
        permute_123: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_61: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_215: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_123, unsqueeze_61)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_58: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_123, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_22: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_58);  slice_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_57: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_123, 3, 0, 128);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_22: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_22, slice_57], -1);  neg_22 = slice_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_62: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_216: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_22, unsqueeze_62);  cat_22 = None
        add_161: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_227: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_320, [1000, 2304])
        permute_124: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_78: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_227, permute_124);  view_227 = permute_124 = None
        view_228: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [1, 1000, 1024]);  mm_78 = None
        view_229: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_228, [1, 1000, -1, 256]);  view_228 = None
        permute_125: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_217: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_125, unsqueeze_61);  unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_60: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_125, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_23: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_60);  slice_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_59: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_125, 3, 0, 128);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_23: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_23, slice_59], -1);  neg_23 = slice_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_218: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_23, unsqueeze_62);  cat_23 = unsqueeze_62 = None
        add_162: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_63: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_162, 2)
        expand_39: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_63, [1, 4, 2, 1000, 256]);  unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_48: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_233: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [1, 8, 1000, 256]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_230: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_320, [1000, 2304]);  convert_element_type_320 = None
        permute_126: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_79: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_230, permute_126);  view_230 = permute_126 = None
        view_231: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [1, 1000, 1024]);  mm_79 = None
        view_232: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_231, [1, 1000, -1, 256]);  view_231 = None
        permute_127: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_64: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_127, 2)
        expand_40: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_64, [1, 4, 2, 1000, 256]);  unsqueeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_49: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_234: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [1, 8, 1000, 256]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_29: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_29, full_default_28);  full_default_29 = full_default_28 = None
        expand_41: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_11, [1, 8, 1000, 1000]);  where_11 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_161, view_233, view_234, expand_41, False, scale = 0.0625);  add_161 = view_233 = view_234 = expand_41 = None
        getitem_44: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_44, [0, 2, 1, 3]);  getitem_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_128, [1, 1000, -1]);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_236: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [1000, 2048]);  view_235 = None
        permute_129: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_80: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_236, permute_129);  view_236 = permute_129 = None
        view_237: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [1, 1000, 2304]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_329: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_46: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_329, 2)
        mean_45: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
        add_163: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_45, 1e-06);  mean_45 = None
        rsqrt_45: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_219: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, rsqrt_45);  convert_element_type_329 = rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_330: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg130_1, torch.float32);  arg130_1 = None
        add_164: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_330, 1.0);  convert_element_type_330 = None
        mul_220: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_219, add_164);  mul_219 = add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_331: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_220, torch.bfloat16);  mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_165: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_158, convert_element_type_331);  add_158 = convert_element_type_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_332: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_165, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_47: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_332, 2)
        mean_46: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        add_166: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_46, 1e-06);  mean_46 = None
        rsqrt_46: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_221: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_332, rsqrt_46);  convert_element_type_332 = rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_333: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg131_1, torch.float32);  arg131_1 = None
        add_167: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_333, 1.0);  convert_element_type_333 = None
        mul_222: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, add_167);  mul_221 = add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_334: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_222, torch.bfloat16);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_238: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_334, [1000, 2304])
        permute_130: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_81: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_238, permute_130);  view_238 = permute_130 = None
        view_239: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [1, 1000, 9216]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_337: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_227: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_337, 0.5)
        mul_223: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_337, convert_element_type_337)
        mul_224: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_223, convert_element_type_337);  mul_223 = None
        mul_225: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, 0.044715);  mul_224 = None
        add_168: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_337, mul_225);  convert_element_type_337 = mul_225 = None
        mul_226: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, 0.7978845608028654);  add_168 = None
        tanh_11: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_226);  mul_226 = None
        add_169: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1);  tanh_11 = None
        mul_228: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, add_169);  mul_227 = add_169 = None
        convert_element_type_338: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_228, torch.bfloat16);  mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_240: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_334, [1000, 2304]);  convert_element_type_334 = None
        permute_131: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_82: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_240, permute_131);  view_240 = permute_131 = None
        view_241: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [1, 1000, 9216]);  mm_82 = None
        mul_229: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_338, view_241);  convert_element_type_338 = view_241 = None
        view_242: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_229, [1000, 9216]);  mul_229 = None
        permute_132: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        mm_83: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_132);  view_242 = permute_132 = None
        view_243: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [1, 1000, 2304]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_343: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_243, torch.float32);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_48: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_343, 2)
        mean_47: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
        add_170: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_47, 1e-06);  mean_47 = None
        rsqrt_47: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_230: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_343, rsqrt_47);  convert_element_type_343 = rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_344: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg135_1, torch.float32);  arg135_1 = None
        add_171: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_344, 1.0);  convert_element_type_344 = None
        mul_231: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, add_171);  mul_230 = add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_345: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_231, torch.bfloat16);  mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_172: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_165, convert_element_type_345);  add_165 = convert_element_type_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_346: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_172, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_49: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_346, 2)
        mean_48: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None
        add_173: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_48, 1e-06);  mean_48 = None
        rsqrt_48: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        mul_232: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_346, rsqrt_48);  convert_element_type_346 = rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_347: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg136_1, torch.float32);  arg136_1 = None
        add_174: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_347, 1.0);  convert_element_type_347 = None
        mul_233: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_232, add_174);  mul_232 = add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_348: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_233, torch.bfloat16);  mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_244: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_348, [1000, 2304])
        permute_133: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_84: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_244, permute_133);  view_244 = permute_133 = None
        view_245: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [1, 1000, 2048]);  mm_84 = None
        view_246: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [1, 1000, -1, 256]);  view_245 = None
        permute_134: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_65: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_234: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_134, unsqueeze_65)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_62: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_134, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_24: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_62);  slice_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_61: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_134, 3, 0, 128);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_24: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_24, slice_61], -1);  neg_24 = slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_66: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_235: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_24, unsqueeze_66);  cat_24 = None
        add_175: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_247: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_348, [1000, 2304])
        permute_135: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        mm_85: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_247, permute_135);  view_247 = permute_135 = None
        view_248: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_85, [1, 1000, 1024]);  mm_85 = None
        view_249: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_248, [1, 1000, -1, 256]);  view_248 = None
        permute_136: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_236: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_136, unsqueeze_65);  unsqueeze_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_64: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_136, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_25: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_64);  slice_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_63: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_136, 3, 0, 128);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_25: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_25, slice_63], -1);  neg_25 = slice_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_237: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_25, unsqueeze_66);  cat_25 = unsqueeze_66 = None
        add_176: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_67: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_176, 2)
        expand_42: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_67, [1, 4, 2, 1000, 256]);  unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_52: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_253: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [1, 8, 1000, 256]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_250: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_348, [1000, 2304]);  convert_element_type_348 = None
        permute_137: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        mm_86: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_250, permute_137);  view_250 = permute_137 = None
        view_251: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [1, 1000, 1024]);  mm_86 = None
        view_252: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [1, 1000, -1, 256]);  view_251 = None
        permute_138: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_68: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_138, 2)
        expand_43: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_68, [1, 4, 2, 1000, 256]);  unsqueeze_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_53: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_254: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [1, 8, 1000, 256]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_32: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_32, full_default_31);  full_default_32 = full_default_31 = None
        expand_44: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_12, [1, 8, 1000, 1000]);  where_12 = None
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_175, view_253, view_254, expand_44, False, scale = 0.0625);  add_175 = view_253 = view_254 = expand_44 = None
        getitem_48: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_48, [0, 2, 1, 3]);  getitem_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_255: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_139, [1, 1000, -1]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_256: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_255, [1000, 2048]);  view_255 = None
        permute_140: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_87: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_256, permute_140);  view_256 = permute_140 = None
        view_257: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [1, 1000, 2304]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_358: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_50: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_358, 2)
        mean_49: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
        add_177: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_49, 1e-06);  mean_49 = None
        rsqrt_49: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        mul_238: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_358, rsqrt_49);  convert_element_type_358 = rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_359: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float32);  arg141_1 = None
        add_178: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_359, 1.0);  convert_element_type_359 = None
        mul_239: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, add_178);  mul_238 = add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_360: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_239, torch.bfloat16);  mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_179: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, convert_element_type_360);  add_172 = convert_element_type_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_361: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_51: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_361, 2)
        mean_50: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        add_180: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_50, 1e-06);  mean_50 = None
        rsqrt_50: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        mul_240: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_361, rsqrt_50);  convert_element_type_361 = rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_362: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg142_1, torch.float32);  arg142_1 = None
        add_181: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_362, 1.0);  convert_element_type_362 = None
        mul_241: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, add_181);  mul_240 = add_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_363: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_258: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_363, [1000, 2304])
        permute_141: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        mm_88: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_258, permute_141);  view_258 = permute_141 = None
        view_259: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [1, 1000, 9216]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_366: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_259, torch.float32);  view_259 = None
        mul_246: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_366, 0.5)
        mul_242: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_366, convert_element_type_366)
        mul_243: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, convert_element_type_366);  mul_242 = None
        mul_244: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, 0.044715);  mul_243 = None
        add_182: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_366, mul_244);  convert_element_type_366 = mul_244 = None
        mul_245: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_182, 0.7978845608028654);  add_182 = None
        tanh_12: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_245);  mul_245 = None
        add_183: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_12, 1);  tanh_12 = None
        mul_247: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, add_183);  mul_246 = add_183 = None
        convert_element_type_367: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_247, torch.bfloat16);  mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_260: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_363, [1000, 2304]);  convert_element_type_363 = None
        permute_142: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        mm_89: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_260, permute_142);  view_260 = permute_142 = None
        view_261: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [1, 1000, 9216]);  mm_89 = None
        mul_248: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_367, view_261);  convert_element_type_367 = view_261 = None
        view_262: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_248, [1000, 9216]);  mul_248 = None
        permute_143: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_90: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_262, permute_143);  view_262 = permute_143 = None
        view_263: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [1, 1000, 2304]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_372: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_52: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_372, 2)
        mean_51: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
        add_184: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_51, 1e-06);  mean_51 = None
        rsqrt_51: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        mul_249: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_372, rsqrt_51);  convert_element_type_372 = rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_373: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg146_1, torch.float32);  arg146_1 = None
        add_185: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_373, 1.0);  convert_element_type_373 = None
        mul_250: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, add_185);  mul_249 = add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_374: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_186: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_179, convert_element_type_374);  add_179 = convert_element_type_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_375: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_186, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_53: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_375, 2)
        mean_52: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None
        add_187: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_52, 1e-06);  mean_52 = None
        rsqrt_52: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
        mul_251: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_375, rsqrt_52);  convert_element_type_375 = rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_376: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg147_1, torch.float32);  arg147_1 = None
        add_188: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_376, 1.0);  convert_element_type_376 = None
        mul_252: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_251, add_188);  mul_251 = add_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_377: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_264: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_377, [1000, 2304])
        permute_144: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        mm_91: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_264, permute_144);  view_264 = permute_144 = None
        view_265: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [1, 1000, 2048]);  mm_91 = None
        view_266: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_265, [1, 1000, -1, 256]);  view_265 = None
        permute_145: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_69: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_253: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_145, unsqueeze_69)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_68: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_145, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_26: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_68);  slice_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_67: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_145, 3, 0, 128);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_26: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_26, slice_67], -1);  neg_26 = slice_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_70: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_254: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_26, unsqueeze_70);  cat_26 = None
        add_189: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_267: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_377, [1000, 2304])
        permute_146: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_92: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_267, permute_146);  view_267 = permute_146 = None
        view_268: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [1, 1000, 1024]);  mm_92 = None
        view_269: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_268, [1, 1000, -1, 256]);  view_268 = None
        permute_147: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_255: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_147, unsqueeze_69);  unsqueeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_70: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_147, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_27: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_70);  slice_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_69: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_147, 3, 0, 128);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_27: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_27, slice_69], -1);  neg_27 = slice_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_256: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_27, unsqueeze_70);  cat_27 = unsqueeze_70 = None
        add_190: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_71: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_190, 2)
        expand_45: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_71, [1, 4, 2, 1000, 256]);  unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_56: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_273: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_56, [1, 8, 1000, 256]);  clone_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_270: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_377, [1000, 2304]);  convert_element_type_377 = None
        permute_148: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        mm_93: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_270, permute_148);  view_270 = permute_148 = None
        view_271: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [1, 1000, 1024]);  mm_93 = None
        view_272: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_271, [1, 1000, -1, 256]);  view_271 = None
        permute_149: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_72: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_149, 2)
        expand_46: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_72, [1, 4, 2, 1000, 256]);  unsqueeze_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_57: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_274: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [1, 8, 1000, 256]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_34: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_34, full_default_33);  full_default_34 = full_default_33 = None
        expand_47: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_13, [1, 8, 1000, 1000]);  where_13 = None
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_189, view_273, view_274, expand_47, False, scale = 0.0625);  add_189 = view_273 = view_274 = expand_47 = None
        getitem_52: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_275: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_150, [1, 1000, -1]);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_276: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_275, [1000, 2048]);  view_275 = None
        permute_151: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_94: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_276, permute_151);  view_276 = permute_151 = None
        view_277: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [1, 1000, 2304]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_386: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_54: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_386, 2)
        mean_53: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
        add_191: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_53, 1e-06);  mean_53 = None
        rsqrt_53: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        mul_257: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_386, rsqrt_53);  convert_element_type_386 = rsqrt_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_387: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg152_1, torch.float32);  arg152_1 = None
        add_192: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_387, 1.0);  convert_element_type_387 = None
        mul_258: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_257, add_192);  mul_257 = add_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_388: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_258, torch.bfloat16);  mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_193: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_186, convert_element_type_388);  add_186 = convert_element_type_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_389: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_193, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_55: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_389, 2)
        mean_54: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        add_194: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_54, 1e-06);  mean_54 = None
        rsqrt_54: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        mul_259: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_389, rsqrt_54);  convert_element_type_389 = rsqrt_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_390: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg153_1, torch.float32);  arg153_1 = None
        add_195: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_390, 1.0);  convert_element_type_390 = None
        mul_260: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, add_195);  mul_259 = add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_391: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_260, torch.bfloat16);  mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_278: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_391, [1000, 2304])
        permute_152: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        mm_95: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_278, permute_152);  view_278 = permute_152 = None
        view_279: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [1, 1000, 9216]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_394: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        mul_265: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_394, 0.5)
        mul_261: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_394, convert_element_type_394)
        mul_262: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, convert_element_type_394);  mul_261 = None
        mul_263: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, 0.044715);  mul_262 = None
        add_196: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_394, mul_263);  convert_element_type_394 = mul_263 = None
        mul_264: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_196, 0.7978845608028654);  add_196 = None
        tanh_13: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_264);  mul_264 = None
        add_197: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_13, 1);  tanh_13 = None
        mul_266: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_265, add_197);  mul_265 = add_197 = None
        convert_element_type_395: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_266, torch.bfloat16);  mul_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_280: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_391, [1000, 2304]);  convert_element_type_391 = None
        permute_153: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_96: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_280, permute_153);  view_280 = permute_153 = None
        view_281: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [1, 1000, 9216]);  mm_96 = None
        mul_267: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_395, view_281);  convert_element_type_395 = view_281 = None
        view_282: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_267, [1000, 9216]);  mul_267 = None
        permute_154: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        mm_97: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_282, permute_154);  view_282 = permute_154 = None
        view_283: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_97, [1, 1000, 2304]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_400: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_56: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_400, 2)
        mean_55: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
        add_198: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_55, 1e-06);  mean_55 = None
        rsqrt_55: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        mul_268: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_400, rsqrt_55);  convert_element_type_400 = rsqrt_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_401: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg157_1, torch.float32);  arg157_1 = None
        add_199: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_401, 1.0);  convert_element_type_401 = None
        mul_269: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_268, add_199);  mul_268 = add_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_402: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_200: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_193, convert_element_type_402);  add_193 = convert_element_type_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_403: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_200, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_57: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_403, 2)
        mean_56: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None
        add_201: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_56, 1e-06);  mean_56 = None
        rsqrt_56: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        mul_270: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, rsqrt_56);  convert_element_type_403 = rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_404: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg158_1, torch.float32);  arg158_1 = None
        add_202: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_404, 1.0);  convert_element_type_404 = None
        mul_271: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, add_202);  mul_270 = add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_405: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_271, torch.bfloat16);  mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_284: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_405, [1000, 2304])
        permute_155: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_98: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_284, permute_155);  view_284 = permute_155 = None
        view_285: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [1, 1000, 2048]);  mm_98 = None
        view_286: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_285, [1, 1000, -1, 256]);  view_285 = None
        permute_156: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_73: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_272: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_156, unsqueeze_73)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_72: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_156, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_28: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_72);  slice_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_71: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_156, 3, 0, 128);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_28: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_28, slice_71], -1);  neg_28 = slice_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_74: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_273: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_28, unsqueeze_74);  cat_28 = None
        add_203: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_287: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_405, [1000, 2304])
        permute_157: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_99: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_287, permute_157);  view_287 = permute_157 = None
        view_288: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_99, [1, 1000, 1024]);  mm_99 = None
        view_289: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_288, [1, 1000, -1, 256]);  view_288 = None
        permute_158: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_274: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_158, unsqueeze_73);  unsqueeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_74: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_158, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_29: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_74);  slice_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_73: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_158, 3, 0, 128);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_29: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_29, slice_73], -1);  neg_29 = slice_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_275: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_29, unsqueeze_74);  cat_29 = unsqueeze_74 = None
        add_204: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_75: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_204, 2)
        expand_48: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_75, [1, 4, 2, 1000, 256]);  unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_60: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_293: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [1, 8, 1000, 256]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_290: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_405, [1000, 2304]);  convert_element_type_405 = None
        permute_159: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        mm_100: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_290, permute_159);  view_290 = permute_159 = None
        view_291: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [1, 1000, 1024]);  mm_100 = None
        view_292: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_291, [1, 1000, -1, 256]);  view_291 = None
        permute_160: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_76: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_160, 2)
        expand_49: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_76, [1, 4, 2, 1000, 256]);  unsqueeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_61: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_294: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [1, 8, 1000, 256]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_37: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_37, full_default_36);  full_default_37 = full_default_36 = None
        expand_50: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_14, [1, 8, 1000, 1000]);  where_14 = None
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_203, view_293, view_294, expand_50, False, scale = 0.0625);  add_203 = view_293 = view_294 = expand_50 = None
        getitem_56: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_295: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_161, [1, 1000, -1]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_296: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [1000, 2048]);  view_295 = None
        permute_162: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        mm_101: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_296, permute_162);  view_296 = permute_162 = None
        view_297: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [1, 1000, 2304]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_415: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_297, torch.float32);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_58: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_415, 2)
        mean_57: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
        add_205: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_57, 1e-06);  mean_57 = None
        rsqrt_57: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        mul_276: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_415, rsqrt_57);  convert_element_type_415 = rsqrt_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_416: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg163_1, torch.float32);  arg163_1 = None
        add_206: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_416, 1.0);  convert_element_type_416 = None
        mul_277: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, add_206);  mul_276 = add_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_417: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_277, torch.bfloat16);  mul_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_207: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_200, convert_element_type_417);  add_200 = convert_element_type_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_418: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_207, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_59: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_418, 2)
        mean_58: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None
        add_208: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_58, 1e-06);  mean_58 = None
        rsqrt_58: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        mul_278: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_418, rsqrt_58);  convert_element_type_418 = rsqrt_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_419: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg164_1, torch.float32);  arg164_1 = None
        add_209: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_419, 1.0);  convert_element_type_419 = None
        mul_279: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, add_209);  mul_278 = add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_420: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_279, torch.bfloat16);  mul_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_298: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_420, [1000, 2304])
        permute_163: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_102: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_298, permute_163);  view_298 = permute_163 = None
        view_299: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [1, 1000, 9216]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_423: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        mul_284: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_423, 0.5)
        mul_280: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_423, convert_element_type_423)
        mul_281: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, convert_element_type_423);  mul_280 = None
        mul_282: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, 0.044715);  mul_281 = None
        add_210: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_423, mul_282);  convert_element_type_423 = mul_282 = None
        mul_283: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_210, 0.7978845608028654);  add_210 = None
        tanh_14: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_283);  mul_283 = None
        add_211: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_14, 1);  tanh_14 = None
        mul_285: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_284, add_211);  mul_284 = add_211 = None
        convert_element_type_424: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_285, torch.bfloat16);  mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_300: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_420, [1000, 2304]);  convert_element_type_420 = None
        permute_164: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        mm_103: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_300, permute_164);  view_300 = permute_164 = None
        view_301: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [1, 1000, 9216]);  mm_103 = None
        mul_286: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_424, view_301);  convert_element_type_424 = view_301 = None
        view_302: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_286, [1000, 9216]);  mul_286 = None
        permute_165: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        mm_104: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_302, permute_165);  view_302 = permute_165 = None
        view_303: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [1, 1000, 2304]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_429: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_60: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_429, 2)
        mean_59: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_60, [-1], True);  pow_60 = None
        add_212: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_59, 1e-06);  mean_59 = None
        rsqrt_59: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        mul_287: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_429, rsqrt_59);  convert_element_type_429 = rsqrt_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_430: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg168_1, torch.float32);  arg168_1 = None
        add_213: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_430, 1.0);  convert_element_type_430 = None
        mul_288: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, add_213);  mul_287 = add_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_431: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_288, torch.bfloat16);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_214: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_207, convert_element_type_431);  add_207 = convert_element_type_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_432: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_61: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_432, 2)
        mean_60: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None
        add_215: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_60, 1e-06);  mean_60 = None
        rsqrt_60: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        mul_289: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_432, rsqrt_60);  convert_element_type_432 = rsqrt_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_433: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg169_1, torch.float32);  arg169_1 = None
        add_216: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_433, 1.0);  convert_element_type_433 = None
        mul_290: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, add_216);  mul_289 = add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_434: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16);  mul_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_304: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_434, [1000, 2304])
        permute_166: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        mm_105: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_304, permute_166);  view_304 = permute_166 = None
        view_305: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_105, [1, 1000, 2048]);  mm_105 = None
        view_306: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_305, [1, 1000, -1, 256]);  view_305 = None
        permute_167: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_77: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_291: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_167, unsqueeze_77)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_78: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_167, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_30: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_78);  slice_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_77: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_167, 3, 0, 128);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_30: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_30, slice_77], -1);  neg_30 = slice_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_78: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_292: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_30, unsqueeze_78);  cat_30 = None
        add_217: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_307: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_434, [1000, 2304])
        permute_168: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_106: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_307, permute_168);  view_307 = permute_168 = None
        view_308: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [1, 1000, 1024]);  mm_106 = None
        view_309: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_308, [1, 1000, -1, 256]);  view_308 = None
        permute_169: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_293: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_169, unsqueeze_77);  unsqueeze_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_80: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_169, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_31: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_80);  slice_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_79: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_169, 3, 0, 128);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_31: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_31, slice_79], -1);  neg_31 = slice_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_294: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_31, unsqueeze_78);  cat_31 = unsqueeze_78 = None
        add_218: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_79: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_218, 2)
        expand_51: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_79, [1, 4, 2, 1000, 256]);  unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_64: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_313: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [1, 8, 1000, 256]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_310: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_434, [1000, 2304]);  convert_element_type_434 = None
        permute_170: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        mm_107: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_310, permute_170);  view_310 = permute_170 = None
        view_311: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_107, [1, 1000, 1024]);  mm_107 = None
        view_312: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_311, [1, 1000, -1, 256]);  view_311 = None
        permute_171: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_80: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_171, 2)
        expand_52: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_80, [1, 4, 2, 1000, 256]);  unsqueeze_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_65: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_314: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1, 8, 1000, 256]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_39: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_39, full_default_38);  full_default_39 = full_default_38 = None
        expand_53: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_15, [1, 8, 1000, 1000]);  where_15 = None
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_217, view_313, view_314, expand_53, False, scale = 0.0625);  add_217 = view_313 = view_314 = expand_53 = None
        getitem_60: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_172: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_315: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_172, [1, 1000, -1]);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_316: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [1000, 2048]);  view_315 = None
        permute_173: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_108: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_316, permute_173);  view_316 = permute_173 = None
        view_317: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [1, 1000, 2304]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_443: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_62: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_443, 2)
        mean_61: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_62, [-1], True);  pow_62 = None
        add_219: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_61, 1e-06);  mean_61 = None
        rsqrt_61: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_295: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_443, rsqrt_61);  convert_element_type_443 = rsqrt_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_444: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg174_1, torch.float32);  arg174_1 = None
        add_220: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_444, 1.0);  convert_element_type_444 = None
        mul_296: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, add_220);  mul_295 = add_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_445: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_221: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_214, convert_element_type_445);  add_214 = convert_element_type_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_446: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_221, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_63: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_446, 2)
        mean_62: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_63, [-1], True);  pow_63 = None
        add_222: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_62, 1e-06);  mean_62 = None
        rsqrt_62: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        mul_297: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_446, rsqrt_62);  convert_element_type_446 = rsqrt_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_447: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg175_1, torch.float32);  arg175_1 = None
        add_223: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_447, 1.0);  convert_element_type_447 = None
        mul_298: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, add_223);  mul_297 = add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_448: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_298, torch.bfloat16);  mul_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_318: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_448, [1000, 2304])
        permute_174: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        mm_109: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_318, permute_174);  view_318 = permute_174 = None
        view_319: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_109, [1, 1000, 9216]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_451: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_319, torch.float32);  view_319 = None
        mul_303: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_451, 0.5)
        mul_299: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_451, convert_element_type_451)
        mul_300: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, convert_element_type_451);  mul_299 = None
        mul_301: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, 0.044715);  mul_300 = None
        add_224: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_451, mul_301);  convert_element_type_451 = mul_301 = None
        mul_302: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_224, 0.7978845608028654);  add_224 = None
        tanh_15: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_302);  mul_302 = None
        add_225: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_15, 1);  tanh_15 = None
        mul_304: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_303, add_225);  mul_303 = add_225 = None
        convert_element_type_452: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_304, torch.bfloat16);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_320: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_448, [1000, 2304]);  convert_element_type_448 = None
        permute_175: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        mm_110: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_320, permute_175);  view_320 = permute_175 = None
        view_321: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [1, 1000, 9216]);  mm_110 = None
        mul_305: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_452, view_321);  convert_element_type_452 = view_321 = None
        view_322: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_305, [1000, 9216]);  mul_305 = None
        permute_176: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_111: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_322, permute_176);  view_322 = permute_176 = None
        view_323: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_111, [1, 1000, 2304]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_457: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_64: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_457, 2)
        mean_63: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_64, [-1], True);  pow_64 = None
        add_226: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_63, 1e-06);  mean_63 = None
        rsqrt_63: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        mul_306: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_457, rsqrt_63);  convert_element_type_457 = rsqrt_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_458: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg179_1, torch.float32);  arg179_1 = None
        add_227: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_458, 1.0);  convert_element_type_458 = None
        mul_307: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, add_227);  mul_306 = add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_459: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_307, torch.bfloat16);  mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_228: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_221, convert_element_type_459);  add_221 = convert_element_type_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_460: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_65: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_460, 2)
        mean_64: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_65, [-1], True);  pow_65 = None
        add_229: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_64, 1e-06);  mean_64 = None
        rsqrt_64: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_308: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_460, rsqrt_64);  convert_element_type_460 = rsqrt_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_461: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg180_1, torch.float32);  arg180_1 = None
        add_230: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_461, 1.0);  convert_element_type_461 = None
        mul_309: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, add_230);  mul_308 = add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_462: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_309, torch.bfloat16);  mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_324: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_462, [1000, 2304])
        permute_177: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        mm_112: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_324, permute_177);  view_324 = permute_177 = None
        view_325: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [1, 1000, 2048]);  mm_112 = None
        view_326: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_325, [1, 1000, -1, 256]);  view_325 = None
        permute_178: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_81: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_310: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_178, unsqueeze_81)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_82: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_178, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_32: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_82);  slice_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_81: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_178, 3, 0, 128);  permute_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_32: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_32, slice_81], -1);  neg_32 = slice_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_82: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_311: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_32, unsqueeze_82);  cat_32 = None
        add_231: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_327: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_462, [1000, 2304])
        permute_179: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_113: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_327, permute_179);  view_327 = permute_179 = None
        view_328: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_113, [1, 1000, 1024]);  mm_113 = None
        view_329: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_328, [1, 1000, -1, 256]);  view_328 = None
        permute_180: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_312: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_180, unsqueeze_81);  unsqueeze_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_84: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_180, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_33: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_84);  slice_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_83: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_180, 3, 0, 128);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_33: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_33, slice_83], -1);  neg_33 = slice_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_313: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_33, unsqueeze_82);  cat_33 = unsqueeze_82 = None
        add_232: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_83: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_232, 2)
        expand_54: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_83, [1, 4, 2, 1000, 256]);  unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_68: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_333: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [1, 8, 1000, 256]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_330: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_462, [1000, 2304]);  convert_element_type_462 = None
        permute_181: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_114: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_330, permute_181);  view_330 = permute_181 = None
        view_331: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [1, 1000, 1024]);  mm_114 = None
        view_332: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_331, [1, 1000, -1, 256]);  view_331 = None
        permute_182: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_84: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_182, 2)
        expand_55: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_84, [1, 4, 2, 1000, 256]);  unsqueeze_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_69: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_334: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [1, 8, 1000, 256]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_42: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_42, full_default_41);  full_default_42 = full_default_41 = None
        expand_56: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_16, [1, 8, 1000, 1000]);  where_16 = None
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_231, view_333, view_334, expand_56, False, scale = 0.0625);  add_231 = view_333 = view_334 = expand_56 = None
        getitem_64: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_183: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_64, [0, 2, 1, 3]);  getitem_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_335: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_183, [1, 1000, -1]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_336: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_335, [1000, 2048]);  view_335 = None
        permute_184: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        mm_115: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_184);  view_336 = permute_184 = None
        view_337: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_115, [1, 1000, 2304]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_472: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_337, torch.float32);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_66: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_472, 2)
        mean_65: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_66, [-1], True);  pow_66 = None
        add_233: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_65, 1e-06);  mean_65 = None
        rsqrt_65: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        mul_314: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_472, rsqrt_65);  convert_element_type_472 = rsqrt_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_473: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg185_1, torch.float32);  arg185_1 = None
        add_234: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_473, 1.0);  convert_element_type_473 = None
        mul_315: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_314, add_234);  mul_314 = add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_474: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_315, torch.bfloat16);  mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_235: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_228, convert_element_type_474);  add_228 = convert_element_type_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_475: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_235, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_67: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_475, 2)
        mean_66: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_67, [-1], True);  pow_67 = None
        add_236: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_66, 1e-06);  mean_66 = None
        rsqrt_66: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_316: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, rsqrt_66);  convert_element_type_475 = rsqrt_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_476: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg186_1, torch.float32);  arg186_1 = None
        add_237: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_476, 1.0);  convert_element_type_476 = None
        mul_317: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_316, add_237);  mul_316 = add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_477: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_317, torch.bfloat16);  mul_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_338: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_477, [1000, 2304])
        permute_185: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        mm_116: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_338, permute_185);  view_338 = permute_185 = None
        view_339: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [1, 1000, 9216]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_480: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        mul_322: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_480, 0.5)
        mul_318: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_480, convert_element_type_480)
        mul_319: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, convert_element_type_480);  mul_318 = None
        mul_320: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_319, 0.044715);  mul_319 = None
        add_238: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_480, mul_320);  convert_element_type_480 = mul_320 = None
        mul_321: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_238, 0.7978845608028654);  add_238 = None
        tanh_16: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_321);  mul_321 = None
        add_239: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_16, 1);  tanh_16 = None
        mul_323: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, add_239);  mul_322 = add_239 = None
        convert_element_type_481: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_323, torch.bfloat16);  mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_340: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_477, [1000, 2304]);  convert_element_type_477 = None
        permute_186: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        mm_117: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_340, permute_186);  view_340 = permute_186 = None
        view_341: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_117, [1, 1000, 9216]);  mm_117 = None
        mul_324: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_481, view_341);  convert_element_type_481 = view_341 = None
        view_342: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_324, [1000, 9216]);  mul_324 = None
        permute_187: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_118: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_342, permute_187);  view_342 = permute_187 = None
        view_343: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [1, 1000, 2304]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_486: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_343, torch.float32);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_68: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_486, 2)
        mean_67: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_68, [-1], True);  pow_68 = None
        add_240: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_67, 1e-06);  mean_67 = None
        rsqrt_67: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        mul_325: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_486, rsqrt_67);  convert_element_type_486 = rsqrt_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_487: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg190_1, torch.float32);  arg190_1 = None
        add_241: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_487, 1.0);  convert_element_type_487 = None
        mul_326: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, add_241);  mul_325 = add_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_488: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_326, torch.bfloat16);  mul_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_242: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_235, convert_element_type_488);  add_235 = convert_element_type_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_489: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_242, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_69: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_489, 2)
        mean_68: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_69, [-1], True);  pow_69 = None
        add_243: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_68, 1e-06);  mean_68 = None
        rsqrt_68: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_327: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_489, rsqrt_68);  convert_element_type_489 = rsqrt_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_490: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg191_1, torch.float32);  arg191_1 = None
        add_244: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_490, 1.0);  convert_element_type_490 = None
        mul_328: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, add_244);  mul_327 = add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_491: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16);  mul_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_344: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_491, [1000, 2304])
        permute_188: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        mm_119: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_344, permute_188);  view_344 = permute_188 = None
        view_345: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_119, [1, 1000, 2048]);  mm_119 = None
        view_346: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [1, 1000, -1, 256]);  view_345 = None
        permute_189: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_85: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_329: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_189, unsqueeze_85)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_88: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_189, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_34: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_88);  slice_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_87: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_189, 3, 0, 128);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_34: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_34, slice_87], -1);  neg_34 = slice_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_86: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_330: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_34, unsqueeze_86);  cat_34 = None
        add_245: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_347: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_491, [1000, 2304])
        permute_190: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        mm_120: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_347, permute_190);  view_347 = permute_190 = None
        view_348: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [1, 1000, 1024]);  mm_120 = None
        view_349: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_348, [1, 1000, -1, 256]);  view_348 = None
        permute_191: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_331: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_191, unsqueeze_85);  unsqueeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_90: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_191, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_35: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_90);  slice_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_89: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_191, 3, 0, 128);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_35: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_35, slice_89], -1);  neg_35 = slice_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_332: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_35, unsqueeze_86);  cat_35 = unsqueeze_86 = None
        add_246: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_331, mul_332);  mul_331 = mul_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_87: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_246, 2)
        expand_57: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_87, [1, 4, 2, 1000, 256]);  unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_72: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_353: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [1, 8, 1000, 256]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_350: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_491, [1000, 2304]);  convert_element_type_491 = None
        permute_192: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        mm_121: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_350, permute_192);  view_350 = permute_192 = None
        view_351: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_121, [1, 1000, 1024]);  mm_121 = None
        view_352: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_351, [1, 1000, -1, 256]);  view_351 = None
        permute_193: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_88: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_193, 2)
        expand_58: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_88, [1, 4, 2, 1000, 256]);  unsqueeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_73: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_354: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [1, 8, 1000, 256]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_44: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_44, full_default_43);  full_default_44 = full_default_43 = None
        expand_59: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_17, [1, 8, 1000, 1000]);  where_17 = None
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_245, view_353, view_354, expand_59, False, scale = 0.0625);  add_245 = view_353 = view_354 = expand_59 = None
        getitem_68: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_194: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3]);  getitem_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_355: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_194, [1, 1000, -1]);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_356: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_355, [1000, 2048]);  view_355 = None
        permute_195: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        mm_122: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_356, permute_195);  view_356 = permute_195 = None
        view_357: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [1, 1000, 2304]);  mm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_500: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_357, torch.float32);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_70: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_500, 2)
        mean_69: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_70, [-1], True);  pow_70 = None
        add_247: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_69, 1e-06);  mean_69 = None
        rsqrt_69: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_333: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_500, rsqrt_69);  convert_element_type_500 = rsqrt_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_501: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg196_1, torch.float32);  arg196_1 = None
        add_248: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_501, 1.0);  convert_element_type_501 = None
        mul_334: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, add_248);  mul_333 = add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_502: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_249: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_242, convert_element_type_502);  add_242 = convert_element_type_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_503: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_71: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_503, 2)
        mean_70: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_71, [-1], True);  pow_71 = None
        add_250: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_70, 1e-06);  mean_70 = None
        rsqrt_70: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        mul_335: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_503, rsqrt_70);  convert_element_type_503 = rsqrt_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_504: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg197_1, torch.float32);  arg197_1 = None
        add_251: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_504, 1.0);  convert_element_type_504 = None
        mul_336: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_335, add_251);  mul_335 = add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_505: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_336, torch.bfloat16);  mul_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_358: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [1000, 2304])
        permute_196: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        mm_123: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_358, permute_196);  view_358 = permute_196 = None
        view_359: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_123, [1, 1000, 9216]);  mm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_508: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        mul_341: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_508, 0.5)
        mul_337: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_508, convert_element_type_508)
        mul_338: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, convert_element_type_508);  mul_337 = None
        mul_339: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, 0.044715);  mul_338 = None
        add_252: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_508, mul_339);  convert_element_type_508 = mul_339 = None
        mul_340: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_252, 0.7978845608028654);  add_252 = None
        tanh_17: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_340);  mul_340 = None
        add_253: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_17, 1);  tanh_17 = None
        mul_342: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_341, add_253);  mul_341 = add_253 = None
        convert_element_type_509: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_342, torch.bfloat16);  mul_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_360: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [1000, 2304]);  convert_element_type_505 = None
        permute_197: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        mm_124: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_360, permute_197);  view_360 = permute_197 = None
        view_361: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [1, 1000, 9216]);  mm_124 = None
        mul_343: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_509, view_361);  convert_element_type_509 = view_361 = None
        view_362: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_343, [1000, 9216]);  mul_343 = None
        permute_198: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        mm_125: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_362, permute_198);  view_362 = permute_198 = None
        view_363: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [1, 1000, 2304]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_514: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_72: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_514, 2)
        mean_71: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_72, [-1], True);  pow_72 = None
        add_254: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_71, 1e-06);  mean_71 = None
        rsqrt_71: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        mul_344: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, rsqrt_71);  convert_element_type_514 = rsqrt_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_515: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg201_1, torch.float32);  arg201_1 = None
        add_255: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_515, 1.0);  convert_element_type_515 = None
        mul_345: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_344, add_255);  mul_344 = add_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_516: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_345, torch.bfloat16);  mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_256: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_249, convert_element_type_516);  add_249 = convert_element_type_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_517: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_256, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_73: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_517, 2)
        mean_72: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_73, [-1], True);  pow_73 = None
        add_257: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_72, 1e-06);  mean_72 = None
        rsqrt_72: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        mul_346: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_517, rsqrt_72);  convert_element_type_517 = rsqrt_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_518: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg202_1, torch.float32);  arg202_1 = None
        add_258: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_518, 1.0);  convert_element_type_518 = None
        mul_347: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, add_258);  mul_346 = add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_519: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_347, torch.bfloat16);  mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_364: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_519, [1000, 2304])
        permute_199: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        mm_126: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_364, permute_199);  view_364 = permute_199 = None
        view_365: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [1, 1000, 2048]);  mm_126 = None
        view_366: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [1, 1000, -1, 256]);  view_365 = None
        permute_200: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_89: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_348: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_200, unsqueeze_89)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_92: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_200, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_36: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_92);  slice_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_91: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_200, 3, 0, 128);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_36: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_36, slice_91], -1);  neg_36 = slice_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_90: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_349: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_36, unsqueeze_90);  cat_36 = None
        add_259: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_348, mul_349);  mul_348 = mul_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_367: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_519, [1000, 2304])
        permute_201: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        mm_127: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_367, permute_201);  view_367 = permute_201 = None
        view_368: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_127, [1, 1000, 1024]);  mm_127 = None
        view_369: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_368, [1, 1000, -1, 256]);  view_368 = None
        permute_202: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_350: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_202, unsqueeze_89);  unsqueeze_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_94: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_202, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_37: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_94);  slice_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_93: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_202, 3, 0, 128);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_37: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_37, slice_93], -1);  neg_37 = slice_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_351: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_37, unsqueeze_90);  cat_37 = unsqueeze_90 = None
        add_260: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_91: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_260, 2)
        expand_60: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_91, [1, 4, 2, 1000, 256]);  unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_76: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_373: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [1, 8, 1000, 256]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_370: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_519, [1000, 2304]);  convert_element_type_519 = None
        permute_203: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        mm_128: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_370, permute_203);  view_370 = permute_203 = None
        view_371: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [1, 1000, 1024]);  mm_128 = None
        view_372: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_371, [1, 1000, -1, 256]);  view_371 = None
        permute_204: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_92: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_204, 2)
        expand_61: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_92, [1, 4, 2, 1000, 256]);  unsqueeze_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_77: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_374: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [1, 8, 1000, 256]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_47: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_47, full_default_46);  full_default_47 = full_default_46 = None
        expand_62: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_18, [1, 8, 1000, 1000]);  where_18 = None
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_259, view_373, view_374, expand_62, False, scale = 0.0625);  add_259 = view_373 = view_374 = expand_62 = None
        getitem_72: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_205: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_375: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_205, [1, 1000, -1]);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_376: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_375, [1000, 2048]);  view_375 = None
        permute_206: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        mm_129: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_376, permute_206);  view_376 = permute_206 = None
        view_377: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_129, [1, 1000, 2304]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_529: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_74: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_529, 2)
        mean_73: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_74, [-1], True);  pow_74 = None
        add_261: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_73, 1e-06);  mean_73 = None
        rsqrt_73: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        mul_352: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_529, rsqrt_73);  convert_element_type_529 = rsqrt_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_530: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg207_1, torch.float32);  arg207_1 = None
        add_262: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_530, 1.0);  convert_element_type_530 = None
        mul_353: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, add_262);  mul_352 = add_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_531: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_353, torch.bfloat16);  mul_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_263: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_256, convert_element_type_531);  add_256 = convert_element_type_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_532: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_263, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_75: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_532, 2)
        mean_74: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_75, [-1], True);  pow_75 = None
        add_264: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_74, 1e-06);  mean_74 = None
        rsqrt_74: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        mul_354: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_532, rsqrt_74);  convert_element_type_532 = rsqrt_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_533: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg208_1, torch.float32);  arg208_1 = None
        add_265: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_533, 1.0);  convert_element_type_533 = None
        mul_355: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_354, add_265);  mul_354 = add_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_534: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_355, torch.bfloat16);  mul_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_378: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_534, [1000, 2304])
        permute_207: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        mm_130: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_378, permute_207);  view_378 = permute_207 = None
        view_379: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [1, 1000, 9216]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_537: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_379, torch.float32);  view_379 = None
        mul_360: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_537, 0.5)
        mul_356: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_537, convert_element_type_537)
        mul_357: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, convert_element_type_537);  mul_356 = None
        mul_358: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, 0.044715);  mul_357 = None
        add_266: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_537, mul_358);  convert_element_type_537 = mul_358 = None
        mul_359: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_266, 0.7978845608028654);  add_266 = None
        tanh_18: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_359);  mul_359 = None
        add_267: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_18, 1);  tanh_18 = None
        mul_361: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, add_267);  mul_360 = add_267 = None
        convert_element_type_538: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_361, torch.bfloat16);  mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_380: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_534, [1000, 2304]);  convert_element_type_534 = None
        permute_208: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        mm_131: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_380, permute_208);  view_380 = permute_208 = None
        view_381: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_131, [1, 1000, 9216]);  mm_131 = None
        mul_362: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_538, view_381);  convert_element_type_538 = view_381 = None
        view_382: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_362, [1000, 9216]);  mul_362 = None
        permute_209: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        mm_132: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_382, permute_209);  view_382 = permute_209 = None
        view_383: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [1, 1000, 2304]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_543: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_383, torch.float32);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_76: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_543, 2)
        mean_75: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_76, [-1], True);  pow_76 = None
        add_268: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_75, 1e-06);  mean_75 = None
        rsqrt_75: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        mul_363: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_543, rsqrt_75);  convert_element_type_543 = rsqrt_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_544: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg212_1, torch.float32);  arg212_1 = None
        add_269: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_544, 1.0);  convert_element_type_544 = None
        mul_364: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_363, add_269);  mul_363 = add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_545: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_364, torch.bfloat16);  mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_270: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_263, convert_element_type_545);  add_263 = convert_element_type_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_546: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_270, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_77: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_546, 2)
        mean_76: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_77, [-1], True);  pow_77 = None
        add_271: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_76, 1e-06);  mean_76 = None
        rsqrt_76: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        mul_365: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_546, rsqrt_76);  convert_element_type_546 = rsqrt_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_547: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg213_1, torch.float32);  arg213_1 = None
        add_272: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_547, 1.0);  convert_element_type_547 = None
        mul_366: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, add_272);  mul_365 = add_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_548: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_366, torch.bfloat16);  mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_384: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_548, [1000, 2304])
        permute_210: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        mm_133: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_384, permute_210);  view_384 = permute_210 = None
        view_385: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_133, [1, 1000, 2048]);  mm_133 = None
        view_386: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_385, [1, 1000, -1, 256]);  view_385 = None
        permute_211: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_93: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_367: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_211, unsqueeze_93)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_98: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_211, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_38: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_98);  slice_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_97: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_211, 3, 0, 128);  permute_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_38: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_38, slice_97], -1);  neg_38 = slice_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_94: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_368: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_38, unsqueeze_94);  cat_38 = None
        add_273: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_387: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_548, [1000, 2304])
        permute_212: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        mm_134: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_387, permute_212);  view_387 = permute_212 = None
        view_388: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [1, 1000, 1024]);  mm_134 = None
        view_389: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_388, [1, 1000, -1, 256]);  view_388 = None
        permute_213: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_369: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_213, unsqueeze_93);  unsqueeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_100: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_213, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_39: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_100);  slice_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_99: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_213, 3, 0, 128);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_39: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_39, slice_99], -1);  neg_39 = slice_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_370: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_39, unsqueeze_94);  cat_39 = unsqueeze_94 = None
        add_274: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_95: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_274, 2)
        expand_63: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_95, [1, 4, 2, 1000, 256]);  unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_80: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_393: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [1, 8, 1000, 256]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_390: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_548, [1000, 2304]);  convert_element_type_548 = None
        permute_214: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        mm_135: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_390, permute_214);  view_390 = permute_214 = None
        view_391: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_135, [1, 1000, 1024]);  mm_135 = None
        view_392: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_391, [1, 1000, -1, 256]);  view_391 = None
        permute_215: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_96: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_215, 2)
        expand_64: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_96, [1, 4, 2, 1000, 256]);  unsqueeze_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_81: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_394: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [1, 8, 1000, 256]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_49: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_49, full_default_48);  full_default_49 = full_default_48 = None
        expand_65: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_19, [1, 8, 1000, 1000]);  where_19 = None
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_273, view_393, view_394, expand_65, False, scale = 0.0625);  add_273 = view_393 = view_394 = expand_65 = None
        getitem_76: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_216: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_76, [0, 2, 1, 3]);  getitem_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_395: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_216, [1, 1000, -1]);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_396: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_395, [1000, 2048]);  view_395 = None
        permute_217: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        mm_136: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_396, permute_217);  view_396 = permute_217 = None
        view_397: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [1, 1000, 2304]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_557: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_397, torch.float32);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_78: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_557, 2)
        mean_77: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_78, [-1], True);  pow_78 = None
        add_275: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_77, 1e-06);  mean_77 = None
        rsqrt_77: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        mul_371: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_557, rsqrt_77);  convert_element_type_557 = rsqrt_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_558: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg218_1, torch.float32);  arg218_1 = None
        add_276: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_558, 1.0);  convert_element_type_558 = None
        mul_372: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, add_276);  mul_371 = add_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_559: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_372, torch.bfloat16);  mul_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_277: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_270, convert_element_type_559);  add_270 = convert_element_type_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_560: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_277, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_79: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_560, 2)
        mean_78: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_79, [-1], True);  pow_79 = None
        add_278: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_78, 1e-06);  mean_78 = None
        rsqrt_78: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
        mul_373: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_560, rsqrt_78);  convert_element_type_560 = rsqrt_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_561: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg219_1, torch.float32);  arg219_1 = None
        add_279: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_561, 1.0);  convert_element_type_561 = None
        mul_374: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_373, add_279);  mul_373 = add_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_562: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_374, torch.bfloat16);  mul_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_398: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_562, [1000, 2304])
        permute_218: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        mm_137: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_398, permute_218);  view_398 = permute_218 = None
        view_399: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [1, 1000, 9216]);  mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_565: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_399, torch.float32);  view_399 = None
        mul_379: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_565, 0.5)
        mul_375: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_565, convert_element_type_565)
        mul_376: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_375, convert_element_type_565);  mul_375 = None
        mul_377: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_376, 0.044715);  mul_376 = None
        add_280: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_565, mul_377);  convert_element_type_565 = mul_377 = None
        mul_378: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_280, 0.7978845608028654);  add_280 = None
        tanh_19: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_378);  mul_378 = None
        add_281: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_19, 1);  tanh_19 = None
        mul_380: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_379, add_281);  mul_379 = add_281 = None
        convert_element_type_566: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_380, torch.bfloat16);  mul_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_400: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_562, [1000, 2304]);  convert_element_type_562 = None
        permute_219: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        mm_138: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_400, permute_219);  view_400 = permute_219 = None
        view_401: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [1, 1000, 9216]);  mm_138 = None
        mul_381: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, view_401);  convert_element_type_566 = view_401 = None
        view_402: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_381, [1000, 9216]);  mul_381 = None
        permute_220: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        mm_139: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_402, permute_220);  view_402 = permute_220 = None
        view_403: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_139, [1, 1000, 2304]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_571: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_403, torch.float32);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_80: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_571, 2)
        mean_79: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_80, [-1], True);  pow_80 = None
        add_282: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_79, 1e-06);  mean_79 = None
        rsqrt_79: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        mul_382: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_571, rsqrt_79);  convert_element_type_571 = rsqrt_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_572: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg223_1, torch.float32);  arg223_1 = None
        add_283: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_572, 1.0);  convert_element_type_572 = None
        mul_383: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_382, add_283);  mul_382 = add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_573: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_383, torch.bfloat16);  mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_284: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_277, convert_element_type_573);  add_277 = convert_element_type_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_574: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_284, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_81: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_574, 2)
        mean_80: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_81, [-1], True);  pow_81 = None
        add_285: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_80, 1e-06);  mean_80 = None
        rsqrt_80: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
        mul_384: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_574, rsqrt_80);  convert_element_type_574 = rsqrt_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_575: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg224_1, torch.float32);  arg224_1 = None
        add_286: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_575, 1.0);  convert_element_type_575 = None
        mul_385: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_384, add_286);  mul_384 = add_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_576: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_385, torch.bfloat16);  mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_404: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_576, [1000, 2304])
        permute_221: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        mm_140: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_404, permute_221);  view_404 = permute_221 = None
        view_405: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [1, 1000, 2048]);  mm_140 = None
        view_406: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_405, [1, 1000, -1, 256]);  view_405 = None
        permute_222: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_97: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_386: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_222, unsqueeze_97)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_102: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_222, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_40: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_102);  slice_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_101: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_222, 3, 0, 128);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_40: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_40, slice_101], -1);  neg_40 = slice_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_98: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_387: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_40, unsqueeze_98);  cat_40 = None
        add_287: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_407: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_576, [1000, 2304])
        permute_223: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        mm_141: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_407, permute_223);  view_407 = permute_223 = None
        view_408: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_141, [1, 1000, 1024]);  mm_141 = None
        view_409: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_408, [1, 1000, -1, 256]);  view_408 = None
        permute_224: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_388: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_224, unsqueeze_97);  unsqueeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_104: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_224, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_41: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_104);  slice_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_103: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_224, 3, 0, 128);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_41: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_41, slice_103], -1);  neg_41 = slice_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_389: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_41, unsqueeze_98);  cat_41 = unsqueeze_98 = None
        add_288: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_388, mul_389);  mul_388 = mul_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_99: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_288, 2)
        expand_66: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_99, [1, 4, 2, 1000, 256]);  unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_84: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_413: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [1, 8, 1000, 256]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_410: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_576, [1000, 2304]);  convert_element_type_576 = None
        permute_225: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        mm_142: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_410, permute_225);  view_410 = permute_225 = None
        view_411: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [1, 1000, 1024]);  mm_142 = None
        view_412: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [1, 1000, -1, 256]);  view_411 = None
        permute_226: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_100: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_226, 2)
        expand_67: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_100, [1, 4, 2, 1000, 256]);  unsqueeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_85: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_414: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [1, 8, 1000, 256]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_52: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_52, full_default_51);  full_default_52 = full_default_51 = None
        expand_68: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_20, [1, 8, 1000, 1000]);  where_20 = None
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_287, view_413, view_414, expand_68, False, scale = 0.0625);  add_287 = view_413 = view_414 = expand_68 = None
        getitem_80: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3]);  getitem_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_415: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_227, [1, 1000, -1]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_416: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_415, [1000, 2048]);  view_415 = None
        permute_228: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        mm_143: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_416, permute_228);  view_416 = permute_228 = None
        view_417: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_143, [1, 1000, 2304]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_586: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_82: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_586, 2)
        mean_81: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_82, [-1], True);  pow_82 = None
        add_289: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_81, 1e-06);  mean_81 = None
        rsqrt_81: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        mul_390: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_586, rsqrt_81);  convert_element_type_586 = rsqrt_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_587: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg229_1, torch.float32);  arg229_1 = None
        add_290: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_587, 1.0);  convert_element_type_587 = None
        mul_391: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, add_290);  mul_390 = add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_588: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_391, torch.bfloat16);  mul_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_291: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_284, convert_element_type_588);  add_284 = convert_element_type_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_589: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_291, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_83: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_589, 2)
        mean_82: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_83, [-1], True);  pow_83 = None
        add_292: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_82, 1e-06);  mean_82 = None
        rsqrt_82: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        mul_392: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_589, rsqrt_82);  convert_element_type_589 = rsqrt_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_590: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg230_1, torch.float32);  arg230_1 = None
        add_293: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_590, 1.0);  convert_element_type_590 = None
        mul_393: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, add_293);  mul_392 = add_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_591: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_393, torch.bfloat16);  mul_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_418: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_591, [1000, 2304])
        permute_229: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        mm_144: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_418, permute_229);  view_418 = permute_229 = None
        view_419: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [1, 1000, 9216]);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_594: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        mul_398: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_594, 0.5)
        mul_394: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_594, convert_element_type_594)
        mul_395: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_394, convert_element_type_594);  mul_394 = None
        mul_396: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, 0.044715);  mul_395 = None
        add_294: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_594, mul_396);  convert_element_type_594 = mul_396 = None
        mul_397: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_294, 0.7978845608028654);  add_294 = None
        tanh_20: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_397);  mul_397 = None
        add_295: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_20, 1);  tanh_20 = None
        mul_399: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_398, add_295);  mul_398 = add_295 = None
        convert_element_type_595: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_399, torch.bfloat16);  mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_420: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_591, [1000, 2304]);  convert_element_type_591 = None
        permute_230: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        mm_145: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_420, permute_230);  view_420 = permute_230 = None
        view_421: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_145, [1, 1000, 9216]);  mm_145 = None
        mul_400: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_595, view_421);  convert_element_type_595 = view_421 = None
        view_422: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_400, [1000, 9216]);  mul_400 = None
        permute_231: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        mm_146: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_422, permute_231);  view_422 = permute_231 = None
        view_423: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [1, 1000, 2304]);  mm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_600: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_423, torch.float32);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_84: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_600, 2)
        mean_83: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_84, [-1], True);  pow_84 = None
        add_296: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_83, 1e-06);  mean_83 = None
        rsqrt_83: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        mul_401: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_600, rsqrt_83);  convert_element_type_600 = rsqrt_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_601: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg234_1, torch.float32);  arg234_1 = None
        add_297: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_601, 1.0);  convert_element_type_601 = None
        mul_402: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_401, add_297);  mul_401 = add_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_602: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_402, torch.bfloat16);  mul_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_298: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_291, convert_element_type_602);  add_291 = convert_element_type_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_603: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_298, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_85: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_603, 2)
        mean_84: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_85, [-1], True);  pow_85 = None
        add_299: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_84, 1e-06);  mean_84 = None
        rsqrt_84: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        mul_403: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_603, rsqrt_84);  convert_element_type_603 = rsqrt_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_604: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg235_1, torch.float32);  arg235_1 = None
        add_300: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_604, 1.0);  convert_element_type_604 = None
        mul_404: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, add_300);  mul_403 = add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_605: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_404, torch.bfloat16);  mul_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_424: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_605, [1000, 2304])
        permute_232: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        mm_147: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_424, permute_232);  view_424 = permute_232 = None
        view_425: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_147, [1, 1000, 2048]);  mm_147 = None
        view_426: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_425, [1, 1000, -1, 256]);  view_425 = None
        permute_233: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_101: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_405: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_233, unsqueeze_101)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_108: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_233, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_42: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_108);  slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_107: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_233, 3, 0, 128);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_42: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_42, slice_107], -1);  neg_42 = slice_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_102: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_406: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_42, unsqueeze_102);  cat_42 = None
        add_301: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_405, mul_406);  mul_405 = mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_427: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_605, [1000, 2304])
        permute_234: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        mm_148: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_427, permute_234);  view_427 = permute_234 = None
        view_428: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_148, [1, 1000, 1024]);  mm_148 = None
        view_429: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_428, [1, 1000, -1, 256]);  view_428 = None
        permute_235: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_429, [0, 2, 1, 3]);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_407: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_235, unsqueeze_101);  unsqueeze_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_110: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_235, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_43: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_110);  slice_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_109: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_235, 3, 0, 128);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_43: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_43, slice_109], -1);  neg_43 = slice_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_408: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_43, unsqueeze_102);  cat_43 = unsqueeze_102 = None
        add_302: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_103: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_302, 2)
        expand_69: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_103, [1, 4, 2, 1000, 256]);  unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_88: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_433: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [1, 8, 1000, 256]);  clone_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_430: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_605, [1000, 2304]);  convert_element_type_605 = None
        permute_236: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        mm_149: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_430, permute_236);  view_430 = permute_236 = None
        view_431: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_149, [1, 1000, 1024]);  mm_149 = None
        view_432: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_431, [1, 1000, -1, 256]);  view_431 = None
        permute_237: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_104: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_237, 2)
        expand_70: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_104, [1, 4, 2, 1000, 256]);  unsqueeze_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_89: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_434: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [1, 8, 1000, 256]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_54: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_54, full_default_53);  full_default_54 = full_default_53 = None
        expand_71: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_21, [1, 8, 1000, 1000]);  where_21 = None
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_301, view_433, view_434, expand_71, False, scale = 0.0625);  add_301 = view_433 = view_434 = expand_71 = None
        getitem_84: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_238: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_84, [0, 2, 1, 3]);  getitem_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_435: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_238, [1, 1000, -1]);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_436: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_435, [1000, 2048]);  view_435 = None
        permute_239: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        mm_150: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_436, permute_239);  view_436 = permute_239 = None
        view_437: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_150, [1, 1000, 2304]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_614: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_86: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_614, 2)
        mean_85: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_86, [-1], True);  pow_86 = None
        add_303: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_85, 1e-06);  mean_85 = None
        rsqrt_85: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        mul_409: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_614, rsqrt_85);  convert_element_type_614 = rsqrt_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_615: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg240_1, torch.float32);  arg240_1 = None
        add_304: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_615, 1.0);  convert_element_type_615 = None
        mul_410: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_409, add_304);  mul_409 = add_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_616: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_410, torch.bfloat16);  mul_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_305: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_298, convert_element_type_616);  add_298 = convert_element_type_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_617: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_305, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_87: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_617, 2)
        mean_86: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_87, [-1], True);  pow_87 = None
        add_306: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_86, 1e-06);  mean_86 = None
        rsqrt_86: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        mul_411: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_617, rsqrt_86);  convert_element_type_617 = rsqrt_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_618: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg241_1, torch.float32);  arg241_1 = None
        add_307: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_618, 1.0);  convert_element_type_618 = None
        mul_412: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, add_307);  mul_411 = add_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_619: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_412, torch.bfloat16);  mul_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_438: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_619, [1000, 2304])
        permute_240: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        mm_151: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_438, permute_240);  view_438 = permute_240 = None
        view_439: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_151, [1, 1000, 9216]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_622: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_439, torch.float32);  view_439 = None
        mul_417: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_622, 0.5)
        mul_413: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_622, convert_element_type_622)
        mul_414: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, convert_element_type_622);  mul_413 = None
        mul_415: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_414, 0.044715);  mul_414 = None
        add_308: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_622, mul_415);  convert_element_type_622 = mul_415 = None
        mul_416: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_308, 0.7978845608028654);  add_308 = None
        tanh_21: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_416);  mul_416 = None
        add_309: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_21, 1);  tanh_21 = None
        mul_418: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_417, add_309);  mul_417 = add_309 = None
        convert_element_type_623: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_418, torch.bfloat16);  mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_440: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_619, [1000, 2304]);  convert_element_type_619 = None
        permute_241: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        mm_152: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_440, permute_241);  view_440 = permute_241 = None
        view_441: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_152, [1, 1000, 9216]);  mm_152 = None
        mul_419: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_623, view_441);  convert_element_type_623 = view_441 = None
        view_442: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_419, [1000, 9216]);  mul_419 = None
        permute_242: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        mm_153: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_442, permute_242);  view_442 = permute_242 = None
        view_443: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_153, [1, 1000, 2304]);  mm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_628: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_443, torch.float32);  view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_88: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_628, 2)
        mean_87: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_88, [-1], True);  pow_88 = None
        add_310: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_87, 1e-06);  mean_87 = None
        rsqrt_87: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        mul_420: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_628, rsqrt_87);  convert_element_type_628 = rsqrt_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_629: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg245_1, torch.float32);  arg245_1 = None
        add_311: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_629, 1.0);  convert_element_type_629 = None
        mul_421: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, add_311);  mul_420 = add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_630: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_421, torch.bfloat16);  mul_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_312: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_305, convert_element_type_630);  add_305 = convert_element_type_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_631: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_312, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_89: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_631, 2)
        mean_88: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_89, [-1], True);  pow_89 = None
        add_313: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_88, 1e-06);  mean_88 = None
        rsqrt_88: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        mul_422: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_631, rsqrt_88);  convert_element_type_631 = rsqrt_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_632: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg246_1, torch.float32);  arg246_1 = None
        add_314: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_632, 1.0);  convert_element_type_632 = None
        mul_423: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_422, add_314);  mul_422 = add_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_633: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_423, torch.bfloat16);  mul_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_444: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_633, [1000, 2304])
        permute_243: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        mm_154: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_444, permute_243);  view_444 = permute_243 = None
        view_445: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_154, [1, 1000, 2048]);  mm_154 = None
        view_446: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_445, [1, 1000, -1, 256]);  view_445 = None
        permute_244: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_105: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_424: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_244, unsqueeze_105)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_112: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_244, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_44: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_112);  slice_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_111: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_244, 3, 0, 128);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_44: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_44, slice_111], -1);  neg_44 = slice_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_106: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_425: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_44, unsqueeze_106);  cat_44 = None
        add_315: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_447: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_633, [1000, 2304])
        permute_245: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        mm_155: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_447, permute_245);  view_447 = permute_245 = None
        view_448: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_155, [1, 1000, 1024]);  mm_155 = None
        view_449: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_448, [1, 1000, -1, 256]);  view_448 = None
        permute_246: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_426: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_246, unsqueeze_105);  unsqueeze_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_114: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_246, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_45: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_114);  slice_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_113: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_246, 3, 0, 128);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_45: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_45, slice_113], -1);  neg_45 = slice_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_427: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_45, unsqueeze_106);  cat_45 = unsqueeze_106 = None
        add_316: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_426, mul_427);  mul_426 = mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_107: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_316, 2)
        expand_72: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_107, [1, 4, 2, 1000, 256]);  unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_92: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_453: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [1, 8, 1000, 256]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_450: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_633, [1000, 2304]);  convert_element_type_633 = None
        permute_247: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        mm_156: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_450, permute_247);  view_450 = permute_247 = None
        view_451: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [1, 1000, 1024]);  mm_156 = None
        view_452: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_451, [1, 1000, -1, 256]);  view_451 = None
        permute_248: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_108: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_248, 2)
        expand_73: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_108, [1, 4, 2, 1000, 256]);  unsqueeze_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_93: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_454: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [1, 8, 1000, 256]);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_57: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_57, full_default_56);  full_default_57 = full_default_56 = None
        expand_74: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_22, [1, 8, 1000, 1000]);  where_22 = None
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_315, view_453, view_454, expand_74, False, scale = 0.0625);  add_315 = view_453 = view_454 = expand_74 = None
        getitem_88: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_249: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_88, [0, 2, 1, 3]);  getitem_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_455: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_249, [1, 1000, -1]);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_456: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_455, [1000, 2048]);  view_455 = None
        permute_250: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        mm_157: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_456, permute_250);  view_456 = permute_250 = None
        view_457: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_157, [1, 1000, 2304]);  mm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_643: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_457, torch.float32);  view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_90: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_643, 2)
        mean_89: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_90, [-1], True);  pow_90 = None
        add_317: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_89, 1e-06);  mean_89 = None
        rsqrt_89: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        mul_428: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_643, rsqrt_89);  convert_element_type_643 = rsqrt_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_644: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg251_1, torch.float32);  arg251_1 = None
        add_318: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_644, 1.0);  convert_element_type_644 = None
        mul_429: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_428, add_318);  mul_428 = add_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_645: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_429, torch.bfloat16);  mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_319: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_312, convert_element_type_645);  add_312 = convert_element_type_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_646: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_319, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_91: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_646, 2)
        mean_90: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_91, [-1], True);  pow_91 = None
        add_320: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_90, 1e-06);  mean_90 = None
        rsqrt_90: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        mul_430: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_646, rsqrt_90);  convert_element_type_646 = rsqrt_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_647: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg252_1, torch.float32);  arg252_1 = None
        add_321: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_647, 1.0);  convert_element_type_647 = None
        mul_431: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, add_321);  mul_430 = add_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_648: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_431, torch.bfloat16);  mul_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_458: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_648, [1000, 2304])
        permute_251: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        mm_158: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_458, permute_251);  view_458 = permute_251 = None
        view_459: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [1, 1000, 9216]);  mm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_651: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        mul_436: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_651, 0.5)
        mul_432: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_651, convert_element_type_651)
        mul_433: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, convert_element_type_651);  mul_432 = None
        mul_434: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_433, 0.044715);  mul_433 = None
        add_322: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_651, mul_434);  convert_element_type_651 = mul_434 = None
        mul_435: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_322, 0.7978845608028654);  add_322 = None
        tanh_22: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_435);  mul_435 = None
        add_323: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_22, 1);  tanh_22 = None
        mul_437: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_436, add_323);  mul_436 = add_323 = None
        convert_element_type_652: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_437, torch.bfloat16);  mul_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_460: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_648, [1000, 2304]);  convert_element_type_648 = None
        permute_252: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        mm_159: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_460, permute_252);  view_460 = permute_252 = None
        view_461: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_159, [1, 1000, 9216]);  mm_159 = None
        mul_438: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_652, view_461);  convert_element_type_652 = view_461 = None
        view_462: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_438, [1000, 9216]);  mul_438 = None
        permute_253: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        mm_160: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_462, permute_253);  view_462 = permute_253 = None
        view_463: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_160, [1, 1000, 2304]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_657: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_463, torch.float32);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_92: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_657, 2)
        mean_91: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_92, [-1], True);  pow_92 = None
        add_324: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_91, 1e-06);  mean_91 = None
        rsqrt_91: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        mul_439: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_657, rsqrt_91);  convert_element_type_657 = rsqrt_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_658: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg256_1, torch.float32);  arg256_1 = None
        add_325: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_658, 1.0);  convert_element_type_658 = None
        mul_440: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_439, add_325);  mul_439 = add_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_659: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_440, torch.bfloat16);  mul_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_326: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_319, convert_element_type_659);  add_319 = convert_element_type_659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_660: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_326, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_93: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_660, 2)
        mean_92: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_93, [-1], True);  pow_93 = None
        add_327: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_92, 1e-06);  mean_92 = None
        rsqrt_92: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        mul_441: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_660, rsqrt_92);  convert_element_type_660 = rsqrt_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_661: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg257_1, torch.float32);  arg257_1 = None
        add_328: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_661, 1.0);  convert_element_type_661 = None
        mul_442: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, add_328);  mul_441 = add_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_662: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_442, torch.bfloat16);  mul_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_464: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_662, [1000, 2304])
        permute_254: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        mm_161: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_464, permute_254);  view_464 = permute_254 = None
        view_465: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_161, [1, 1000, 2048]);  mm_161 = None
        view_466: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_465, [1, 1000, -1, 256]);  view_465 = None
        permute_255: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_109: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_443: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_255, unsqueeze_109)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_118: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_255, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_46: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_118);  slice_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_117: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_255, 3, 0, 128);  permute_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_46: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_46, slice_117], -1);  neg_46 = slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_110: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_444: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_46, unsqueeze_110);  cat_46 = None
        add_329: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_467: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_662, [1000, 2304])
        permute_256: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        mm_162: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_467, permute_256);  view_467 = permute_256 = None
        view_468: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_162, [1, 1000, 1024]);  mm_162 = None
        view_469: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_468, [1, 1000, -1, 256]);  view_468 = None
        permute_257: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_445: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_257, unsqueeze_109);  unsqueeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_120: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_257, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_47: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_120);  slice_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_119: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_257, 3, 0, 128);  permute_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_47: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_47, slice_119], -1);  neg_47 = slice_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_446: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_47, unsqueeze_110);  cat_47 = unsqueeze_110 = None
        add_330: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_111: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_330, 2)
        expand_75: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_111, [1, 4, 2, 1000, 256]);  unsqueeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_96: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_473: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [1, 8, 1000, 256]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_470: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_662, [1000, 2304]);  convert_element_type_662 = None
        permute_258: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        mm_163: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_470, permute_258);  view_470 = permute_258 = None
        view_471: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_163, [1, 1000, 1024]);  mm_163 = None
        view_472: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_471, [1, 1000, -1, 256]);  view_471 = None
        permute_259: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_112: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_259, 2)
        expand_76: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_112, [1, 4, 2, 1000, 256]);  unsqueeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_97: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_474: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [1, 8, 1000, 256]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_59: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_59, full_default_58);  full_default_59 = full_default_58 = None
        expand_77: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_23, [1, 8, 1000, 1000]);  where_23 = None
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_329, view_473, view_474, expand_77, False, scale = 0.0625);  add_329 = view_473 = view_474 = expand_77 = None
        getitem_92: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_260: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_92, [0, 2, 1, 3]);  getitem_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_475: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_260, [1, 1000, -1]);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_476: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_475, [1000, 2048]);  view_475 = None
        permute_261: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        mm_164: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_476, permute_261);  view_476 = permute_261 = None
        view_477: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_164, [1, 1000, 2304]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_671: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_477, torch.float32);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_94: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_671, 2)
        mean_93: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_94, [-1], True);  pow_94 = None
        add_331: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_93, 1e-06);  mean_93 = None
        rsqrt_93: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        mul_447: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_671, rsqrt_93);  convert_element_type_671 = rsqrt_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_672: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg262_1, torch.float32);  arg262_1 = None
        add_332: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_672, 1.0);  convert_element_type_672 = None
        mul_448: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_447, add_332);  mul_447 = add_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_673: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_448, torch.bfloat16);  mul_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_333: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_326, convert_element_type_673);  add_326 = convert_element_type_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_674: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_333, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_95: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_674, 2)
        mean_94: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_95, [-1], True);  pow_95 = None
        add_334: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_94, 1e-06);  mean_94 = None
        rsqrt_94: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        mul_449: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_674, rsqrt_94);  convert_element_type_674 = rsqrt_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_675: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg263_1, torch.float32);  arg263_1 = None
        add_335: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_675, 1.0);  convert_element_type_675 = None
        mul_450: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_449, add_335);  mul_449 = add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_676: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_450, torch.bfloat16);  mul_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_478: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_676, [1000, 2304])
        permute_262: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        mm_165: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_478, permute_262);  view_478 = permute_262 = None
        view_479: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_165, [1, 1000, 9216]);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_679: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_479, torch.float32);  view_479 = None
        mul_455: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_679, 0.5)
        mul_451: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_679, convert_element_type_679)
        mul_452: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_451, convert_element_type_679);  mul_451 = None
        mul_453: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_452, 0.044715);  mul_452 = None
        add_336: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_679, mul_453);  convert_element_type_679 = mul_453 = None
        mul_454: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_336, 0.7978845608028654);  add_336 = None
        tanh_23: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_454);  mul_454 = None
        add_337: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_23, 1);  tanh_23 = None
        mul_456: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_455, add_337);  mul_455 = add_337 = None
        convert_element_type_680: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_456, torch.bfloat16);  mul_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_480: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_676, [1000, 2304]);  convert_element_type_676 = None
        permute_263: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        mm_166: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_480, permute_263);  view_480 = permute_263 = None
        view_481: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_166, [1, 1000, 9216]);  mm_166 = None
        mul_457: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_680, view_481);  convert_element_type_680 = view_481 = None
        view_482: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_457, [1000, 9216]);  mul_457 = None
        permute_264: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        mm_167: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_482, permute_264);  view_482 = permute_264 = None
        view_483: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_167, [1, 1000, 2304]);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_685: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_483, torch.float32);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_96: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_685, 2)
        mean_95: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_96, [-1], True);  pow_96 = None
        add_338: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_95, 1e-06);  mean_95 = None
        rsqrt_95: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        mul_458: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_685, rsqrt_95);  convert_element_type_685 = rsqrt_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_686: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg267_1, torch.float32);  arg267_1 = None
        add_339: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_686, 1.0);  convert_element_type_686 = None
        mul_459: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_458, add_339);  mul_458 = add_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_687: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_459, torch.bfloat16);  mul_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_340: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_333, convert_element_type_687);  add_333 = convert_element_type_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_688: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_340, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_97: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_688, 2)
        mean_96: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_97, [-1], True);  pow_97 = None
        add_341: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_96, 1e-06);  mean_96 = None
        rsqrt_96: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        mul_460: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_688, rsqrt_96);  convert_element_type_688 = rsqrt_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_689: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg268_1, torch.float32);  arg268_1 = None
        add_342: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_689, 1.0);  convert_element_type_689 = None
        mul_461: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_460, add_342);  mul_460 = add_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_690: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_461, torch.bfloat16);  mul_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_484: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_690, [1000, 2304])
        permute_265: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        mm_168: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_484, permute_265);  view_484 = permute_265 = None
        view_485: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_168, [1, 1000, 2048]);  mm_168 = None
        view_486: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [1, 1000, -1, 256]);  view_485 = None
        permute_266: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_113: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_462: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_266, unsqueeze_113)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_122: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_266, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_48: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_122);  slice_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_121: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_266, 3, 0, 128);  permute_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_48: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_48, slice_121], -1);  neg_48 = slice_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_114: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_463: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_48, unsqueeze_114);  cat_48 = None
        add_343: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_462, mul_463);  mul_462 = mul_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_487: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_690, [1000, 2304])
        permute_267: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        mm_169: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_487, permute_267);  view_487 = permute_267 = None
        view_488: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_169, [1, 1000, 1024]);  mm_169 = None
        view_489: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_488, [1, 1000, -1, 256]);  view_488 = None
        permute_268: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_464: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_268, unsqueeze_113);  unsqueeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_124: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_268, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_49: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_124);  slice_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_123: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_268, 3, 0, 128);  permute_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_49: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_49, slice_123], -1);  neg_49 = slice_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_465: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_49, unsqueeze_114);  cat_49 = unsqueeze_114 = None
        add_344: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_115: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_344, 2)
        expand_78: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_115, [1, 4, 2, 1000, 256]);  unsqueeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_100: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_493: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [1, 8, 1000, 256]);  clone_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_490: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_690, [1000, 2304]);  convert_element_type_690 = None
        permute_269: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        mm_170: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_490, permute_269);  view_490 = permute_269 = None
        view_491: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [1, 1000, 1024]);  mm_170 = None
        view_492: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_491, [1, 1000, -1, 256]);  view_491 = None
        permute_270: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_116: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_270, 2)
        expand_79: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_116, [1, 4, 2, 1000, 256]);  unsqueeze_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_101: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_494: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [1, 8, 1000, 256]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_62: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_62, full_default_61);  expand_1 = full_default_62 = full_default_61 = None
        expand_80: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_24, [1, 8, 1000, 1000]);  where_24 = None
        _scaled_dot_product_efficient_attention_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_343, view_493, view_494, expand_80, False, scale = 0.0625);  add_343 = view_493 = view_494 = expand_80 = None
        getitem_96: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_24[0];  _scaled_dot_product_efficient_attention_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_271: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_96, [0, 2, 1, 3]);  getitem_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_495: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_271, [1, 1000, -1]);  permute_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_496: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_495, [1000, 2048]);  view_495 = None
        permute_272: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        mm_171: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_496, permute_272);  view_496 = permute_272 = None
        view_497: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_171, [1, 1000, 2304]);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_700: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_497, torch.float32);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_98: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_700, 2)
        mean_97: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_98, [-1], True);  pow_98 = None
        add_345: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_97, 1e-06);  mean_97 = None
        rsqrt_97: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
        mul_466: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_700, rsqrt_97);  convert_element_type_700 = rsqrt_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_701: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg273_1, torch.float32);  arg273_1 = None
        add_346: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_701, 1.0);  convert_element_type_701 = None
        mul_467: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, add_346);  mul_466 = add_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_702: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_467, torch.bfloat16);  mul_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_347: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_340, convert_element_type_702);  add_340 = convert_element_type_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_703: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_347, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_99: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_703, 2)
        mean_98: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_99, [-1], True);  pow_99 = None
        add_348: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_98, 1e-06);  mean_98 = None
        rsqrt_98: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
        mul_468: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_703, rsqrt_98);  convert_element_type_703 = rsqrt_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_704: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg274_1, torch.float32);  arg274_1 = None
        add_349: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_704, 1.0);  convert_element_type_704 = None
        mul_469: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_468, add_349);  mul_468 = add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_705: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_469, torch.bfloat16);  mul_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_498: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_705, [1000, 2304])
        permute_273: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        mm_172: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_498, permute_273);  view_498 = permute_273 = None
        view_499: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_172, [1, 1000, 9216]);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_708: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_499, torch.float32);  view_499 = None
        mul_474: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, 0.5)
        mul_470: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, convert_element_type_708)
        mul_471: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_470, convert_element_type_708);  mul_470 = None
        mul_472: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_471, 0.044715);  mul_471 = None
        add_350: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_708, mul_472);  convert_element_type_708 = mul_472 = None
        mul_473: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_350, 0.7978845608028654);  add_350 = None
        tanh_24: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_473);  mul_473 = None
        add_351: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_24, 1);  tanh_24 = None
        mul_475: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_474, add_351);  mul_474 = add_351 = None
        convert_element_type_709: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_475, torch.bfloat16);  mul_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_500: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_705, [1000, 2304]);  convert_element_type_705 = None
        permute_274: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        mm_173: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_500, permute_274);  view_500 = permute_274 = None
        view_501: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_173, [1, 1000, 9216]);  mm_173 = None
        mul_476: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_709, view_501);  convert_element_type_709 = view_501 = None
        view_502: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_476, [1000, 9216]);  mul_476 = None
        permute_275: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        mm_174: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_502, permute_275);  view_502 = permute_275 = None
        view_503: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_174, [1, 1000, 2304]);  mm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_714: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_100: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_714, 2)
        mean_99: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_100, [-1], True);  pow_100 = None
        add_352: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_99, 1e-06);  mean_99 = None
        rsqrt_99: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_352);  add_352 = None
        mul_477: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_714, rsqrt_99);  convert_element_type_714 = rsqrt_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_715: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg278_1, torch.float32);  arg278_1 = None
        add_353: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_715, 1.0);  convert_element_type_715 = None
        mul_478: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_477, add_353);  mul_477 = add_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_716: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_478, torch.bfloat16);  mul_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_354: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_347, convert_element_type_716);  add_347 = convert_element_type_716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_717: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_354, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_101: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_717, 2)
        mean_100: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_101, [-1], True);  pow_101 = None
        add_355: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_100, 1e-06);  mean_100 = None
        rsqrt_100: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        mul_479: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_717, rsqrt_100);  convert_element_type_717 = rsqrt_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_718: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg279_1, torch.float32);  arg279_1 = None
        add_356: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_718, 1.0);  convert_element_type_718 = None
        mul_480: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_479, add_356);  mul_479 = add_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_719: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_480, torch.bfloat16);  mul_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:270 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_504: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_719, [1000, 2304])
        permute_276: "bf16[2304, 2048][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        mm_175: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_504, permute_276);  view_504 = permute_276 = None
        view_505: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_175, [1, 1000, 2048]);  mm_175 = None
        view_506: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_505, [1, 1000, -1, 256]);  view_505 = None
        permute_277: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:176 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_117: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_481: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_277, unsqueeze_117)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_128: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_277, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_50: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.neg.default(slice_128);  slice_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_127: "bf16[1, 8, 1000, 128][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_277, 3, 0, 128);  permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_50: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_50, slice_127], -1);  neg_50 = slice_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:177 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_118: "bf16[1, 1, 1000, 256][256000, 256000, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:178 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_482: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_50, unsqueeze_118);  cat_50 = None
        add_357: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:271 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_507: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_719, [1000, 2304])
        permute_278: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        mm_176: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_507, permute_278);  view_507 = permute_278 = None
        view_508: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_176, [1, 1000, 1024]);  mm_176 = None
        view_509: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_508, [1, 1000, -1, 256]);  view_508 = None
        permute_279: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_483: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_279, unsqueeze_117);  unsqueeze_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:153 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_130: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_279, 3, 128, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_51: "bf16[1, 4, 1000, 128][512000, 128, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_130);  slice_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:152 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_129: "bf16[1, 4, 1000, 128][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_279, 3, 0, 128);  permute_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:154 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_51: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_51, slice_129], -1);  neg_51 = slice_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:179 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_484: "bf16[1, 4, 1000, 256][1024000, 256000, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_51, unsqueeze_118);  cat_51 = unsqueeze_118 = None
        add_358: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_119: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_358, 2)
        expand_81: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_119, [1, 4, 2, 1000, 256]);  unsqueeze_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_104: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_513: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_104, [1, 8, 1000, 256]);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:272 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_510: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_719, [1000, 2304]);  convert_element_type_719 = None
        permute_280: "bf16[2304, 1024][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        mm_177: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_510, permute_280);  view_510 = permute_280 = None
        view_511: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_177, [1, 1000, 1024]);  mm_177 = None
        view_512: "bf16[1, 1000, 4, 256][1024000, 1024, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [1, 1000, -1, 256]);  view_511 = None
        permute_281: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_120: "bf16[1, 4, 1, 1000, 256][1024000, 256, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_281, 2)
        expand_82: "bf16[1, 4, 2, 1000, 256][1024000, 256, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_120, [1, 4, 2, 1000, 256]);  unsqueeze_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_105: "bf16[1, 4, 2, 1000, 256][2048000, 512000, 256000, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_514: "bf16[1, 8, 1000, 256][2048000, 256000, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [1, 8, 1000, 256]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_64: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_64, full_default_63);  expand = full_default_64 = full_default_63 = None
        expand_83: "bf16[1, 8, 1000, 1000][1000000, 0, 1000, 1]cuda:0" = torch.ops.aten.expand.default(where_25, [1, 8, 1000, 1000]);  where_25 = None
        _scaled_dot_product_efficient_attention_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(add_357, view_513, view_514, expand_83, False, scale = 0.0625);  add_357 = view_513 = view_514 = expand_83 = None
        getitem_100: "bf16[1, 8, 1000, 256][2048000, 256, 2048, 1]cuda:0" = _scaled_dot_product_efficient_attention_25[0];  _scaled_dot_product_efficient_attention_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_6: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_6, 2, -4095, 9223372036854775807);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_5: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_8, 2, -4095, 9223372036854775807);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_5: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_16: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_28, 2, -4095, 9223372036854775807);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_15: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_36, 2, -4095, 9223372036854775807);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_10: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_26: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_50, 2, -4095, 9223372036854775807);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_25: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_64, 2, -4095, 9223372036854775807);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_15: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_36: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_72, 2, -4095, 9223372036854775807);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_35: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_92, 2, -4095, 9223372036854775807);  add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_20: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_46: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_94, 2, -4095, 9223372036854775807);  permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_45: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_120, 2, -4095, 9223372036854775807);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_25: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_56: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_116, 2, -4095, 9223372036854775807);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_55: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_148, 2, -4095, 9223372036854775807);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_30: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_66: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_138, 2, -4095, 9223372036854775807);  permute_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_65: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_176, 2, -4095, 9223372036854775807);  add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_35: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_76: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_160, 2, -4095, 9223372036854775807);  permute_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_75: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_204, 2, -4095, 9223372036854775807);  add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_40: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_86: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_182, 2, -4095, 9223372036854775807);  permute_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_85: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_232, 2, -4095, 9223372036854775807);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_45: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_96: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_204, 2, -4095, 9223372036854775807);  permute_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_95: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_260, 2, -4095, 9223372036854775807);  add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_50: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_106: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_226, 2, -4095, 9223372036854775807);  permute_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_105: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_288, 2, -4095, 9223372036854775807);  add_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_55: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_116: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_248, 2, -4095, 9223372036854775807);  permute_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_115: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_316, 2, -4095, 9223372036854775807);  add_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_60: "i64[][]cuda:0" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_126: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_270, 2, -4095, 9223372036854775807);  permute_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_125: "bf16[1, 4, 1000, 256][1024000, 256, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_344, 2, -4095, 9223372036854775807);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_282: "bf16[1, 1000, 8, 256][2048000, 2048, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:297 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_515: "bf16[1, 1000, 2048][2048000, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(permute_282, [1, 1000, -1]);  permute_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:298 in forward, code: attn_output = self.o_proj(attn_output)
        view_516: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_515, [1000, 2048]);  view_515 = None
        permute_283: "bf16[2048, 2304][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        mm_178: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_516, permute_283);  view_516 = permute_283 = None
        view_517: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_178, [1, 1000, 2304]);  mm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_728: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_517, torch.float32);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_102: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_728, 2)
        mean_101: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_102, [-1], True);  pow_102 = None
        add_359: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_101, 1e-06);  mean_101 = None
        rsqrt_101: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
        mul_485: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_728, rsqrt_101);  convert_element_type_728 = rsqrt_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_729: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg284_1, torch.float32);  arg284_1 = None
        add_360: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_729, 1.0);  convert_element_type_729 = None
        mul_486: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, add_360);  mul_485 = add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_730: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_486, torch.bfloat16);  mul_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:338 in forward, code: hidden_states = residual + hidden_states
        add_361: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_354, convert_element_type_730);  add_354 = convert_element_type_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_731: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_361, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_103: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_731, 2)
        mean_102: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_103, [-1], True);  pow_103 = None
        add_362: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_102, 1e-06);  mean_102 = None
        rsqrt_102: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_362);  add_362 = None
        mul_487: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, rsqrt_102);  convert_element_type_731 = rsqrt_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_732: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg285_1, torch.float32);  arg285_1 = None
        add_363: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_732, 1.0);  convert_element_type_732 = None
        mul_488: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_487, add_363);  mul_487 = add_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_733: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_488, torch.bfloat16);  mul_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_518: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_733, [1000, 2304])
        permute_284: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        mm_179: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_518, permute_284);  view_518 = permute_284 = None
        view_519: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_179, [1, 1000, 9216]);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_736: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_519, torch.float32);  view_519 = None
        mul_493: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_736, 0.5)
        mul_489: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_736, convert_element_type_736)
        mul_490: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_489, convert_element_type_736);  mul_489 = None
        mul_491: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_490, 0.044715);  mul_490 = None
        add_364: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_736, mul_491);  convert_element_type_736 = mul_491 = None
        mul_492: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_364, 0.7978845608028654);  add_364 = None
        tanh_25: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.tanh.default(mul_492);  mul_492 = None
        add_365: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_25, 1);  tanh_25 = None
        mul_494: "f32[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_493, add_365);  mul_493 = add_365 = None
        convert_element_type_737: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_494, torch.bfloat16);  mul_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:81 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_520: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_733, [1000, 2304]);  convert_element_type_733 = None
        permute_285: "bf16[2304, 9216][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        mm_180: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(view_520, permute_285);  view_520 = permute_285 = None
        view_521: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [1, 1000, 9216]);  mm_180 = None
        mul_495: "bf16[1, 1000, 9216][9216000, 9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_737, view_521);  convert_element_type_737 = view_521 = None
        view_522: "bf16[1000, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(mul_495, [1000, 9216]);  mul_495 = None
        permute_286: "bf16[9216, 2304][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        mm_181: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_522, permute_286);  view_522 = permute_286 = None
        view_523: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_181, [1, 1000, 2304]);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_742: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_523, torch.float32);  view_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_104: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_742, 2)
        mean_103: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_104, [-1], True);  pow_104 = None
        add_366: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_103, 1e-06);  mean_103 = None
        rsqrt_103: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_366);  add_366 = None
        mul_496: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_742, rsqrt_103);  convert_element_type_742 = rsqrt_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_743: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg289_1, torch.float32);  arg289_1 = None
        add_367: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_743, 1.0);  convert_element_type_743 = None
        mul_497: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_496, add_367);  mul_496 = add_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_744: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_497, torch.bfloat16);  mul_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:344 in forward, code: hidden_states = residual + hidden_states
        add_368: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(add_361, convert_element_type_744);  add_361 = convert_element_type_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:59 in forward, code: output = self._norm(x.float())
        convert_element_type_745: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_368, torch.float32);  add_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:56 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_105: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_745, 2)
        mean_104: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_105, [-1], True);  pow_105 = None
        add_369: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_104, 1e-06);  mean_104 = None
        rsqrt_104: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
        mul_498: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_745, rsqrt_104);  convert_element_type_745 = rsqrt_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:62 in forward, code: output = output * (1.0 + self.weight.float())
        convert_element_type_746: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg290_1, torch.float32);  arg290_1 = None
        add_370: "f32[2304][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_746, 1.0);  convert_element_type_746 = None
        mul_499: "f32[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_498, add_370);  mul_498 = add_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:63 in forward, code: return output.type_as(x)
        convert_element_type_747: "bf16[1, 1000, 2304][2304000, 2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_499, torch.bfloat16);  mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:536 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_524: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_747, [1000, 2304]);  convert_element_type_747 = None
        permute_287: "bf16[2304, 256000][1, 2304]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_182: "bf16[1000, 256000][256000, 1]cuda:0" = torch.ops.aten.mm.default(view_524, permute_287);  view_524 = permute_287 = None
        view_525: "bf16[1, 1000, 256000][256000000, 256000, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [1, 1000, 256000]);  mm_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:538 in forward, code: logits = logits / self.config.final_logit_softcapping
        div: "bf16[1, 1000, 256000][256000000, 256000, 1]cuda:0" = torch.ops.aten.div.Tensor(view_525, 30.0);  view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:539 in forward, code: logits = torch.tanh(logits)
        tanh_26: "bf16[1, 1000, 256000][256000000, 256000, 1]cuda:0" = torch.ops.aten.tanh.default(div);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:540 in forward, code: logits = logits * self.config.final_logit_softcapping
        mul_500: "bf16[1, 1000, 256000][256000000, 256000, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_26, 30.0);  tanh_26 = None
        return (full_default, slice_6, slice_5, permute_17, add_22, full_default_5, slice_16, slice_15, permute_39, add_50, full_default_10, slice_26, slice_25, permute_61, add_78, full_default_15, slice_36, slice_35, permute_83, add_106, full_default_20, slice_46, slice_45, permute_105, add_134, full_default_25, slice_56, slice_55, permute_127, add_162, full_default_30, slice_66, slice_65, permute_149, add_190, full_default_35, slice_76, slice_75, permute_171, add_218, full_default_40, slice_86, slice_85, permute_193, add_246, full_default_45, slice_96, slice_95, permute_215, add_274, full_default_50, slice_106, slice_105, permute_237, add_302, full_default_55, slice_116, slice_115, permute_259, add_330, full_default_60, slice_126, slice_125, permute_281, add_358, mul_500)

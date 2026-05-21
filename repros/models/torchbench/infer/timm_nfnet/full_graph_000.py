class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[128, 3, 192, 192]", arg1_1: "f16[16, 3, 3, 3]", arg2_1: "f16[16, 1, 1, 1]", arg3_1: "f16[16]", arg4_1: "f16[32, 16, 3, 3]", arg5_1: "f16[32, 1, 1, 1]", arg6_1: "f16[32]", arg7_1: "f16[64, 32, 3, 3]", arg8_1: "f16[64, 1, 1, 1]", arg9_1: "f16[64]", arg10_1: "f16[128, 64, 3, 3]", arg11_1: "f16[128, 1, 1, 1]", arg12_1: "f16[128]", arg13_1: "f16[256, 128, 1, 1]", arg14_1: "f16[256, 1, 1, 1]", arg15_1: "f16[256]", arg16_1: "f16[128, 128, 1, 1]", arg17_1: "f16[128, 1, 1, 1]", arg18_1: "f16[128]", arg19_1: "f16[128, 128, 3, 3]", arg20_1: "f16[128, 1, 1, 1]", arg21_1: "f16[128]", arg22_1: "f16[128, 128, 3, 3]", arg23_1: "f16[128, 1, 1, 1]", arg24_1: "f16[128]", arg25_1: "f16[256, 128, 1, 1]", arg26_1: "f16[256, 1, 1, 1]", arg27_1: "f16[256]", arg28_1: "f16[128, 256, 1, 1]", arg29_1: "f16[128]", arg30_1: "f16[256, 128, 1, 1]", arg31_1: "f16[256]", arg32_1: "f16[]", arg33_1: "f16[512, 256, 1, 1]", arg34_1: "f16[512, 1, 1, 1]", arg35_1: "f16[512]", arg36_1: "f16[256, 256, 1, 1]", arg37_1: "f16[256, 1, 1, 1]", arg38_1: "f16[256]", arg39_1: "f16[256, 128, 3, 3]", arg40_1: "f16[256, 1, 1, 1]", arg41_1: "f16[256]", arg42_1: "f16[256, 128, 3, 3]", arg43_1: "f16[256, 1, 1, 1]", arg44_1: "f16[256]", arg45_1: "f16[512, 256, 1, 1]", arg46_1: "f16[512, 1, 1, 1]", arg47_1: "f16[512]", arg48_1: "f16[256, 512, 1, 1]", arg49_1: "f16[256]", arg50_1: "f16[512, 256, 1, 1]", arg51_1: "f16[512]", arg52_1: "f16[]", arg53_1: "f16[256, 512, 1, 1]", arg54_1: "f16[256, 1, 1, 1]", arg55_1: "f16[256]", arg56_1: "f16[256, 128, 3, 3]", arg57_1: "f16[256, 1, 1, 1]", arg58_1: "f16[256]", arg59_1: "f16[256, 128, 3, 3]", arg60_1: "f16[256, 1, 1, 1]", arg61_1: "f16[256]", arg62_1: "f16[512, 256, 1, 1]", arg63_1: "f16[512, 1, 1, 1]", arg64_1: "f16[512]", arg65_1: "f16[256, 512, 1, 1]", arg66_1: "f16[256]", arg67_1: "f16[512, 256, 1, 1]", arg68_1: "f16[512]", arg69_1: "f16[]", arg70_1: "f16[1536, 512, 1, 1]", arg71_1: "f16[1536, 1, 1, 1]", arg72_1: "f16[1536]", arg73_1: "f16[768, 512, 1, 1]", arg74_1: "f16[768, 1, 1, 1]", arg75_1: "f16[768]", arg76_1: "f16[768, 128, 3, 3]", arg77_1: "f16[768, 1, 1, 1]", arg78_1: "f16[768]", arg79_1: "f16[768, 128, 3, 3]", arg80_1: "f16[768, 1, 1, 1]", arg81_1: "f16[768]", arg82_1: "f16[1536, 768, 1, 1]", arg83_1: "f16[1536, 1, 1, 1]", arg84_1: "f16[1536]", arg85_1: "f16[768, 1536, 1, 1]", arg86_1: "f16[768]", arg87_1: "f16[1536, 768, 1, 1]", arg88_1: "f16[1536]", arg89_1: "f16[]", arg90_1: "f16[768, 1536, 1, 1]", arg91_1: "f16[768, 1, 1, 1]", arg92_1: "f16[768]", arg93_1: "f16[768, 128, 3, 3]", arg94_1: "f16[768, 1, 1, 1]", arg95_1: "f16[768]", arg96_1: "f16[768, 128, 3, 3]", arg97_1: "f16[768, 1, 1, 1]", arg98_1: "f16[768]", arg99_1: "f16[1536, 768, 1, 1]", arg100_1: "f16[1536, 1, 1, 1]", arg101_1: "f16[1536]", arg102_1: "f16[768, 1536, 1, 1]", arg103_1: "f16[768]", arg104_1: "f16[1536, 768, 1, 1]", arg105_1: "f16[1536]", arg106_1: "f16[]", arg107_1: "f16[768, 1536, 1, 1]", arg108_1: "f16[768, 1, 1, 1]", arg109_1: "f16[768]", arg110_1: "f16[768, 128, 3, 3]", arg111_1: "f16[768, 1, 1, 1]", arg112_1: "f16[768]", arg113_1: "f16[768, 128, 3, 3]", arg114_1: "f16[768, 1, 1, 1]", arg115_1: "f16[768]", arg116_1: "f16[1536, 768, 1, 1]", arg117_1: "f16[1536, 1, 1, 1]", arg118_1: "f16[1536]", arg119_1: "f16[768, 1536, 1, 1]", arg120_1: "f16[768]", arg121_1: "f16[1536, 768, 1, 1]", arg122_1: "f16[1536]", arg123_1: "f16[]", arg124_1: "f16[768, 1536, 1, 1]", arg125_1: "f16[768, 1, 1, 1]", arg126_1: "f16[768]", arg127_1: "f16[768, 128, 3, 3]", arg128_1: "f16[768, 1, 1, 1]", arg129_1: "f16[768]", arg130_1: "f16[768, 128, 3, 3]", arg131_1: "f16[768, 1, 1, 1]", arg132_1: "f16[768]", arg133_1: "f16[1536, 768, 1, 1]", arg134_1: "f16[1536, 1, 1, 1]", arg135_1: "f16[1536]", arg136_1: "f16[768, 1536, 1, 1]", arg137_1: "f16[768]", arg138_1: "f16[1536, 768, 1, 1]", arg139_1: "f16[1536]", arg140_1: "f16[]", arg141_1: "f16[768, 1536, 1, 1]", arg142_1: "f16[768, 1, 1, 1]", arg143_1: "f16[768]", arg144_1: "f16[768, 128, 3, 3]", arg145_1: "f16[768, 1, 1, 1]", arg146_1: "f16[768]", arg147_1: "f16[768, 128, 3, 3]", arg148_1: "f16[768, 1, 1, 1]", arg149_1: "f16[768]", arg150_1: "f16[1536, 768, 1, 1]", arg151_1: "f16[1536, 1, 1, 1]", arg152_1: "f16[1536]", arg153_1: "f16[768, 1536, 1, 1]", arg154_1: "f16[768]", arg155_1: "f16[1536, 768, 1, 1]", arg156_1: "f16[1536]", arg157_1: "f16[]", arg158_1: "f16[768, 1536, 1, 1]", arg159_1: "f16[768, 1, 1, 1]", arg160_1: "f16[768]", arg161_1: "f16[768, 128, 3, 3]", arg162_1: "f16[768, 1, 1, 1]", arg163_1: "f16[768]", arg164_1: "f16[768, 128, 3, 3]", arg165_1: "f16[768, 1, 1, 1]", arg166_1: "f16[768]", arg167_1: "f16[1536, 768, 1, 1]", arg168_1: "f16[1536, 1, 1, 1]", arg169_1: "f16[1536]", arg170_1: "f16[768, 1536, 1, 1]", arg171_1: "f16[768]", arg172_1: "f16[1536, 768, 1, 1]", arg173_1: "f16[1536]", arg174_1: "f16[]", arg175_1: "f16[1536, 1536, 1, 1]", arg176_1: "f16[1536, 1, 1, 1]", arg177_1: "f16[1536]", arg178_1: "f16[768, 1536, 1, 1]", arg179_1: "f16[768, 1, 1, 1]", arg180_1: "f16[768]", arg181_1: "f16[768, 128, 3, 3]", arg182_1: "f16[768, 1, 1, 1]", arg183_1: "f16[768]", arg184_1: "f16[768, 128, 3, 3]", arg185_1: "f16[768, 1, 1, 1]", arg186_1: "f16[768]", arg187_1: "f16[1536, 768, 1, 1]", arg188_1: "f16[1536, 1, 1, 1]", arg189_1: "f16[1536]", arg190_1: "f16[768, 1536, 1, 1]", arg191_1: "f16[768]", arg192_1: "f16[1536, 768, 1, 1]", arg193_1: "f16[1536]", arg194_1: "f16[]", arg195_1: "f16[768, 1536, 1, 1]", arg196_1: "f16[768, 1, 1, 1]", arg197_1: "f16[768]", arg198_1: "f16[768, 128, 3, 3]", arg199_1: "f16[768, 1, 1, 1]", arg200_1: "f16[768]", arg201_1: "f16[768, 128, 3, 3]", arg202_1: "f16[768, 1, 1, 1]", arg203_1: "f16[768]", arg204_1: "f16[1536, 768, 1, 1]", arg205_1: "f16[1536, 1, 1, 1]", arg206_1: "f16[1536]", arg207_1: "f16[768, 1536, 1, 1]", arg208_1: "f16[768]", arg209_1: "f16[1536, 768, 1, 1]", arg210_1: "f16[1536]", arg211_1: "f16[]", arg212_1: "f16[768, 1536, 1, 1]", arg213_1: "f16[768, 1, 1, 1]", arg214_1: "f16[768]", arg215_1: "f16[768, 128, 3, 3]", arg216_1: "f16[768, 1, 1, 1]", arg217_1: "f16[768]", arg218_1: "f16[768, 128, 3, 3]", arg219_1: "f16[768, 1, 1, 1]", arg220_1: "f16[768]", arg221_1: "f16[1536, 768, 1, 1]", arg222_1: "f16[1536, 1, 1, 1]", arg223_1: "f16[1536]", arg224_1: "f16[768, 1536, 1, 1]", arg225_1: "f16[768]", arg226_1: "f16[1536, 768, 1, 1]", arg227_1: "f16[1536]", arg228_1: "f16[]", arg229_1: "f16[3072, 1536, 1, 1]", arg230_1: "f16[3072, 1, 1, 1]", arg231_1: "f16[3072]", arg232_1: "f16[1000, 3072]", arg233_1: "f16[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view: "f16[1, 16, 27]" = torch.ops.aten.reshape.default(arg1_1, [1, 16, -1]);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type: "f32[1, 16, 27]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 16, 1]" = var_mean[0]
        getitem_1: "f32[1, 16, 1]" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_3: "f16[1, 32, 144]" = torch.ops.aten.reshape.default(arg4_1, [1, 32, -1]);  arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_4: "f32[1, 32, 144]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [0, 2], correction = 0, keepdim = True);  convert_element_type_4 = None
        getitem_2: "f32[1, 32, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 32, 1]" = var_mean_1[1];  var_mean_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_6: "f16[1, 64, 288]" = torch.ops.aten.reshape.default(arg7_1, [1, 64, -1]);  arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_8: "f32[1, 64, 288]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [0, 2], correction = 0, keepdim = True);  convert_element_type_8 = None
        getitem_4: "f32[1, 64, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1]" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_9: "f16[1, 128, 576]" = torch.ops.aten.reshape.default(arg10_1, [1, 128, -1]);  arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_12: "f32[1, 128, 576]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_12, [0, 2], correction = 0, keepdim = True);  convert_element_type_12 = None
        getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_12: "f16[1, 256, 128]" = torch.ops.aten.reshape.default(arg13_1, [1, 256, -1]);  arg13_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_16: "f32[1, 256, 128]" = torch.ops.prims.convert_element_type.default(view_12, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_16, [0, 2], correction = 0, keepdim = True);  convert_element_type_16 = None
        getitem_8: "f32[1, 256, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 256, 1]" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_15: "f16[1, 128, 128]" = torch.ops.aten.reshape.default(arg16_1, [1, 128, -1]);  arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_18: "f32[1, 128, 128]" = torch.ops.prims.convert_element_type.default(view_15, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_18, [0, 2], correction = 0, keepdim = True);  convert_element_type_18 = None
        getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_18: "f16[1, 128, 1152]" = torch.ops.aten.reshape.default(arg19_1, [1, 128, -1]);  arg19_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_22: "f32[1, 128, 1152]" = torch.ops.prims.convert_element_type.default(view_18, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_22, [0, 2], correction = 0, keepdim = True);  convert_element_type_22 = None
        getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_21: "f16[1, 128, 1152]" = torch.ops.aten.reshape.default(arg22_1, [1, 128, -1]);  arg22_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_26: "f32[1, 128, 1152]" = torch.ops.prims.convert_element_type.default(view_21, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_26, [0, 2], correction = 0, keepdim = True);  convert_element_type_26 = None
        getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_24: "f16[1, 256, 128]" = torch.ops.aten.reshape.default(arg25_1, [1, 256, -1]);  arg25_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_30: "f32[1, 256, 128]" = torch.ops.prims.convert_element_type.default(view_24, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_30, [0, 2], correction = 0, keepdim = True);  convert_element_type_30 = None
        getitem_16: "f32[1, 256, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 256, 1]" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_27: "f16[1, 512, 256]" = torch.ops.aten.reshape.default(arg33_1, [1, 512, -1]);  arg33_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_34: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(view_27, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_34, [0, 2], correction = 0, keepdim = True);  convert_element_type_34 = None
        getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_30: "f16[1, 256, 256]" = torch.ops.aten.reshape.default(arg36_1, [1, 256, -1]);  arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_36: "f32[1, 256, 256]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_36, [0, 2], correction = 0, keepdim = True);  convert_element_type_36 = None
        getitem_20: "f32[1, 256, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 256, 1]" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_33: "f16[1, 256, 1152]" = torch.ops.aten.reshape.default(arg39_1, [1, 256, -1]);  arg39_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_40: "f32[1, 256, 1152]" = torch.ops.prims.convert_element_type.default(view_33, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_40, [0, 2], correction = 0, keepdim = True);  convert_element_type_40 = None
        getitem_22: "f32[1, 256, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 256, 1]" = var_mean_11[1];  var_mean_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_36: "f16[1, 256, 1152]" = torch.ops.aten.reshape.default(arg42_1, [1, 256, -1]);  arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_44: "f32[1, 256, 1152]" = torch.ops.prims.convert_element_type.default(view_36, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_44, [0, 2], correction = 0, keepdim = True);  convert_element_type_44 = None
        getitem_24: "f32[1, 256, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 256, 1]" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_39: "f16[1, 512, 256]" = torch.ops.aten.reshape.default(arg45_1, [1, 512, -1]);  arg45_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_48: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_48, [0, 2], correction = 0, keepdim = True);  convert_element_type_48 = None
        getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_42: "f16[1, 256, 512]" = torch.ops.aten.reshape.default(arg53_1, [1, 256, -1]);  arg53_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_52: "f32[1, 256, 512]" = torch.ops.prims.convert_element_type.default(view_42, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_52, [0, 2], correction = 0, keepdim = True);  convert_element_type_52 = None
        getitem_28: "f32[1, 256, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 256, 1]" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_45: "f16[1, 256, 1152]" = torch.ops.aten.reshape.default(arg56_1, [1, 256, -1]);  arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_56: "f32[1, 256, 1152]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_56, [0, 2], correction = 0, keepdim = True);  convert_element_type_56 = None
        getitem_30: "f32[1, 256, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 256, 1]" = var_mean_15[1];  var_mean_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_48: "f16[1, 256, 1152]" = torch.ops.aten.reshape.default(arg59_1, [1, 256, -1]);  arg59_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_60: "f32[1, 256, 1152]" = torch.ops.prims.convert_element_type.default(view_48, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_60, [0, 2], correction = 0, keepdim = True);  convert_element_type_60 = None
        getitem_32: "f32[1, 256, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 256, 1]" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_51: "f16[1, 512, 256]" = torch.ops.aten.reshape.default(arg62_1, [1, 512, -1]);  arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_64: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(view_51, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_64, [0, 2], correction = 0, keepdim = True);  convert_element_type_64 = None
        getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_54: "f16[1, 1536, 512]" = torch.ops.aten.reshape.default(arg70_1, [1, 1536, -1]);  arg70_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_68: "f32[1, 1536, 512]" = torch.ops.prims.convert_element_type.default(view_54, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_68, [0, 2], correction = 0, keepdim = True);  convert_element_type_68 = None
        getitem_36: "f32[1, 1536, 1]" = var_mean_18[0]
        getitem_37: "f32[1, 1536, 1]" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_57: "f16[1, 768, 512]" = torch.ops.aten.reshape.default(arg73_1, [1, 768, -1]);  arg73_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_70: "f32[1, 768, 512]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_70, [0, 2], correction = 0, keepdim = True);  convert_element_type_70 = None
        getitem_38: "f32[1, 768, 1]" = var_mean_19[0]
        getitem_39: "f32[1, 768, 1]" = var_mean_19[1];  var_mean_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_60: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg76_1, [1, 768, -1]);  arg76_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_74: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_60, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_74, [0, 2], correction = 0, keepdim = True);  convert_element_type_74 = None
        getitem_40: "f32[1, 768, 1]" = var_mean_20[0]
        getitem_41: "f32[1, 768, 1]" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_63: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg79_1, [1, 768, -1]);  arg79_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_78: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_63, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_78, [0, 2], correction = 0, keepdim = True);  convert_element_type_78 = None
        getitem_42: "f32[1, 768, 1]" = var_mean_21[0]
        getitem_43: "f32[1, 768, 1]" = var_mean_21[1];  var_mean_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_66: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg82_1, [1, 1536, -1]);  arg82_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_82: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_82, [0, 2], correction = 0, keepdim = True);  convert_element_type_82 = None
        getitem_44: "f32[1, 1536, 1]" = var_mean_22[0]
        getitem_45: "f32[1, 1536, 1]" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_69: "f16[1, 768, 1536]" = torch.ops.aten.reshape.default(arg90_1, [1, 768, -1]);  arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_86: "f32[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_86, [0, 2], correction = 0, keepdim = True);  convert_element_type_86 = None
        getitem_46: "f32[1, 768, 1]" = var_mean_23[0]
        getitem_47: "f32[1, 768, 1]" = var_mean_23[1];  var_mean_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_72: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg93_1, [1, 768, -1]);  arg93_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_90: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_72, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_90, [0, 2], correction = 0, keepdim = True);  convert_element_type_90 = None
        getitem_48: "f32[1, 768, 1]" = var_mean_24[0]
        getitem_49: "f32[1, 768, 1]" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_75: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg96_1, [1, 768, -1]);  arg96_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_94: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_75, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_94, [0, 2], correction = 0, keepdim = True);  convert_element_type_94 = None
        getitem_50: "f32[1, 768, 1]" = var_mean_25[0]
        getitem_51: "f32[1, 768, 1]" = var_mean_25[1];  var_mean_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_78: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg99_1, [1, 1536, -1]);  arg99_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_98: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_78, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_98, [0, 2], correction = 0, keepdim = True);  convert_element_type_98 = None
        getitem_52: "f32[1, 1536, 1]" = var_mean_26[0]
        getitem_53: "f32[1, 1536, 1]" = var_mean_26[1];  var_mean_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_81: "f16[1, 768, 1536]" = torch.ops.aten.reshape.default(arg107_1, [1, 768, -1]);  arg107_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_102: "f32[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(view_81, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_102, [0, 2], correction = 0, keepdim = True);  convert_element_type_102 = None
        getitem_54: "f32[1, 768, 1]" = var_mean_27[0]
        getitem_55: "f32[1, 768, 1]" = var_mean_27[1];  var_mean_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_84: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg110_1, [1, 768, -1]);  arg110_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_106: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_84, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_106, [0, 2], correction = 0, keepdim = True);  convert_element_type_106 = None
        getitem_56: "f32[1, 768, 1]" = var_mean_28[0]
        getitem_57: "f32[1, 768, 1]" = var_mean_28[1];  var_mean_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_87: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg113_1, [1, 768, -1]);  arg113_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_110: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_87, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_110, [0, 2], correction = 0, keepdim = True);  convert_element_type_110 = None
        getitem_58: "f32[1, 768, 1]" = var_mean_29[0]
        getitem_59: "f32[1, 768, 1]" = var_mean_29[1];  var_mean_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_90: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg116_1, [1, 1536, -1]);  arg116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_114: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_90, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_114, [0, 2], correction = 0, keepdim = True);  convert_element_type_114 = None
        getitem_60: "f32[1, 1536, 1]" = var_mean_30[0]
        getitem_61: "f32[1, 1536, 1]" = var_mean_30[1];  var_mean_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_93: "f16[1, 768, 1536]" = torch.ops.aten.reshape.default(arg124_1, [1, 768, -1]);  arg124_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_118: "f32[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_118, [0, 2], correction = 0, keepdim = True);  convert_element_type_118 = None
        getitem_62: "f32[1, 768, 1]" = var_mean_31[0]
        getitem_63: "f32[1, 768, 1]" = var_mean_31[1];  var_mean_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_96: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg127_1, [1, 768, -1]);  arg127_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_122: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_96, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_122, [0, 2], correction = 0, keepdim = True);  convert_element_type_122 = None
        getitem_64: "f32[1, 768, 1]" = var_mean_32[0]
        getitem_65: "f32[1, 768, 1]" = var_mean_32[1];  var_mean_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_99: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg130_1, [1, 768, -1]);  arg130_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_126: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_126, [0, 2], correction = 0, keepdim = True);  convert_element_type_126 = None
        getitem_66: "f32[1, 768, 1]" = var_mean_33[0]
        getitem_67: "f32[1, 768, 1]" = var_mean_33[1];  var_mean_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_102: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg133_1, [1, 1536, -1]);  arg133_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_130: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_102, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_130, [0, 2], correction = 0, keepdim = True);  convert_element_type_130 = None
        getitem_68: "f32[1, 1536, 1]" = var_mean_34[0]
        getitem_69: "f32[1, 1536, 1]" = var_mean_34[1];  var_mean_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_105: "f16[1, 768, 1536]" = torch.ops.aten.reshape.default(arg141_1, [1, 768, -1]);  arg141_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_134: "f32[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_134, [0, 2], correction = 0, keepdim = True);  convert_element_type_134 = None
        getitem_70: "f32[1, 768, 1]" = var_mean_35[0]
        getitem_71: "f32[1, 768, 1]" = var_mean_35[1];  var_mean_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_108: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg144_1, [1, 768, -1]);  arg144_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_138: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_108, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_138, [0, 2], correction = 0, keepdim = True);  convert_element_type_138 = None
        getitem_72: "f32[1, 768, 1]" = var_mean_36[0]
        getitem_73: "f32[1, 768, 1]" = var_mean_36[1];  var_mean_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_111: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg147_1, [1, 768, -1]);  arg147_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_142: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_111, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_142, [0, 2], correction = 0, keepdim = True);  convert_element_type_142 = None
        getitem_74: "f32[1, 768, 1]" = var_mean_37[0]
        getitem_75: "f32[1, 768, 1]" = var_mean_37[1];  var_mean_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_114: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg150_1, [1, 1536, -1]);  arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_146: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_114, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_146, [0, 2], correction = 0, keepdim = True);  convert_element_type_146 = None
        getitem_76: "f32[1, 1536, 1]" = var_mean_38[0]
        getitem_77: "f32[1, 1536, 1]" = var_mean_38[1];  var_mean_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_117: "f16[1, 768, 1536]" = torch.ops.aten.reshape.default(arg158_1, [1, 768, -1]);  arg158_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_150: "f32[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(view_117, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_150, [0, 2], correction = 0, keepdim = True);  convert_element_type_150 = None
        getitem_78: "f32[1, 768, 1]" = var_mean_39[0]
        getitem_79: "f32[1, 768, 1]" = var_mean_39[1];  var_mean_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_120: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg161_1, [1, 768, -1]);  arg161_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_154: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_120, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_154, [0, 2], correction = 0, keepdim = True);  convert_element_type_154 = None
        getitem_80: "f32[1, 768, 1]" = var_mean_40[0]
        getitem_81: "f32[1, 768, 1]" = var_mean_40[1];  var_mean_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_123: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg164_1, [1, 768, -1]);  arg164_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_158: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_123, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_158, [0, 2], correction = 0, keepdim = True);  convert_element_type_158 = None
        getitem_82: "f32[1, 768, 1]" = var_mean_41[0]
        getitem_83: "f32[1, 768, 1]" = var_mean_41[1];  var_mean_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_126: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg167_1, [1, 1536, -1]);  arg167_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_162: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_126, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_162, [0, 2], correction = 0, keepdim = True);  convert_element_type_162 = None
        getitem_84: "f32[1, 1536, 1]" = var_mean_42[0]
        getitem_85: "f32[1, 1536, 1]" = var_mean_42[1];  var_mean_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_129: "f16[1, 1536, 1536]" = torch.ops.aten.reshape.default(arg175_1, [1, 1536, -1]);  arg175_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_166: "f32[1, 1536, 1536]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_166, [0, 2], correction = 0, keepdim = True);  convert_element_type_166 = None
        getitem_86: "f32[1, 1536, 1]" = var_mean_43[0]
        getitem_87: "f32[1, 1536, 1]" = var_mean_43[1];  var_mean_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_132: "f16[1, 768, 1536]" = torch.ops.aten.reshape.default(arg178_1, [1, 768, -1]);  arg178_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_168: "f32[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(view_132, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_168, [0, 2], correction = 0, keepdim = True);  convert_element_type_168 = None
        getitem_88: "f32[1, 768, 1]" = var_mean_44[0]
        getitem_89: "f32[1, 768, 1]" = var_mean_44[1];  var_mean_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_135: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg181_1, [1, 768, -1]);  arg181_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_172: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_135, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_172, [0, 2], correction = 0, keepdim = True);  convert_element_type_172 = None
        getitem_90: "f32[1, 768, 1]" = var_mean_45[0]
        getitem_91: "f32[1, 768, 1]" = var_mean_45[1];  var_mean_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_138: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg184_1, [1, 768, -1]);  arg184_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_176: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_138, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_176, [0, 2], correction = 0, keepdim = True);  convert_element_type_176 = None
        getitem_92: "f32[1, 768, 1]" = var_mean_46[0]
        getitem_93: "f32[1, 768, 1]" = var_mean_46[1];  var_mean_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_141: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg187_1, [1, 1536, -1]);  arg187_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_180: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_141, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_180, [0, 2], correction = 0, keepdim = True);  convert_element_type_180 = None
        getitem_94: "f32[1, 1536, 1]" = var_mean_47[0]
        getitem_95: "f32[1, 1536, 1]" = var_mean_47[1];  var_mean_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_144: "f16[1, 768, 1536]" = torch.ops.aten.reshape.default(arg195_1, [1, 768, -1]);  arg195_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_184: "f32[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(view_144, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_184, [0, 2], correction = 0, keepdim = True);  convert_element_type_184 = None
        getitem_96: "f32[1, 768, 1]" = var_mean_48[0]
        getitem_97: "f32[1, 768, 1]" = var_mean_48[1];  var_mean_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_147: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg198_1, [1, 768, -1]);  arg198_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_188: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_147, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_188, [0, 2], correction = 0, keepdim = True);  convert_element_type_188 = None
        getitem_98: "f32[1, 768, 1]" = var_mean_49[0]
        getitem_99: "f32[1, 768, 1]" = var_mean_49[1];  var_mean_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_150: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg201_1, [1, 768, -1]);  arg201_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_192: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_150, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_192, [0, 2], correction = 0, keepdim = True);  convert_element_type_192 = None
        getitem_100: "f32[1, 768, 1]" = var_mean_50[0]
        getitem_101: "f32[1, 768, 1]" = var_mean_50[1];  var_mean_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_153: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg204_1, [1, 1536, -1]);  arg204_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_196: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_153, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_196, [0, 2], correction = 0, keepdim = True);  convert_element_type_196 = None
        getitem_102: "f32[1, 1536, 1]" = var_mean_51[0]
        getitem_103: "f32[1, 1536, 1]" = var_mean_51[1];  var_mean_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_156: "f16[1, 768, 1536]" = torch.ops.aten.reshape.default(arg212_1, [1, 768, -1]);  arg212_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_200: "f32[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(view_156, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_200, [0, 2], correction = 0, keepdim = True);  convert_element_type_200 = None
        getitem_104: "f32[1, 768, 1]" = var_mean_52[0]
        getitem_105: "f32[1, 768, 1]" = var_mean_52[1];  var_mean_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_159: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg215_1, [1, 768, -1]);  arg215_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_204: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_204, [0, 2], correction = 0, keepdim = True);  convert_element_type_204 = None
        getitem_106: "f32[1, 768, 1]" = var_mean_53[0]
        getitem_107: "f32[1, 768, 1]" = var_mean_53[1];  var_mean_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_162: "f16[1, 768, 1152]" = torch.ops.aten.reshape.default(arg218_1, [1, 768, -1]);  arg218_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_208: "f32[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_208, [0, 2], correction = 0, keepdim = True);  convert_element_type_208 = None
        getitem_108: "f32[1, 768, 1]" = var_mean_54[0]
        getitem_109: "f32[1, 768, 1]" = var_mean_54[1];  var_mean_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_165: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg221_1, [1, 1536, -1]);  arg221_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_212: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_165, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_212, [0, 2], correction = 0, keepdim = True);  convert_element_type_212 = None
        getitem_110: "f32[1, 1536, 1]" = var_mean_55[0]
        getitem_111: "f32[1, 1536, 1]" = var_mean_55[1];  var_mean_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_168: "f16[1, 3072, 1536]" = torch.ops.aten.reshape.default(arg229_1, [1, 3072, -1]);  arg229_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_214: "f32[1, 3072, 1536]" = torch.ops.prims.convert_element_type.default(view_168, torch.float32)
        var_mean_56 = torch.ops.aten.var_mean.correction(convert_element_type_214, [0, 2], correction = 0, keepdim = True);  convert_element_type_214 = None
        getitem_112: "f32[1, 3072, 1]" = var_mean_56[0]
        getitem_113: "f32[1, 3072, 1]" = var_mean_56[1];  var_mean_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f16[128, 3, 193, 193]" = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1, 0, 1], 0.0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        add: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul_1: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul: "f16[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg2_1, 0.19245008972987526);  arg2_1 = None
        view_1: "f16[16]" = torch.ops.aten.reshape.default(mul, [-1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze: "f16[16, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
        mul_2: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze);  mul_1 = unsqueeze = None
        convert_element_type_1: "f16[1, 16, 27]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_2: "f16[16, 3, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_1, [16, 3, 3, 3]);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution: "f16[128, 16, 96, 96]" = torch.ops.aten.convolution.default(constant_pad_nd, view_2, arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  constant_pad_nd = view_2 = arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_2: "f32[128, 16, 96, 96]" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        mul_3: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_4: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf: "f32[128, 16, 96, 96]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_1: "f32[128, 16, 96, 96]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = add_1 = None
        convert_element_type_3: "f16[128, 16, 96, 96]" = torch.ops.prims.convert_element_type.default(mul_5, torch.float16);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_6: "f16[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.7015043497085571);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_1: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, getitem_3);  view_3 = getitem_3 = None
        add_2: "f32[1, 32, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[1, 32, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_8: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_7: "f16[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg5_1, 0.08333333333333333);  arg5_1 = None
        view_4: "f16[32]" = torch.ops.aten.reshape.default(mul_7, [-1]);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_1: "f16[32, 1]" = torch.ops.aten.unsqueeze.default(view_4, -1);  view_4 = None
        mul_9: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_1);  mul_8 = unsqueeze_1 = None
        convert_element_type_5: "f16[1, 32, 144]" = torch.ops.prims.convert_element_type.default(mul_9, torch.float16);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_5: "f16[32, 16, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_5, [32, 16, 3, 3]);  convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_1: "f16[128, 32, 96, 96]" = torch.ops.aten.convolution.default(mul_6, view_5, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_6 = view_5 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_6: "f32[128, 32, 96, 96]" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        mul_10: "f32[128, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_11: "f32[128, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf_1: "f32[128, 32, 96, 96]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_3: "f32[128, 32, 96, 96]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_12: "f32[128, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_10, add_3);  mul_10 = add_3 = None
        convert_element_type_7: "f16[128, 32, 96, 96]" = torch.ops.prims.convert_element_type.default(mul_12, torch.float16);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_13: "f16[128, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.7015043497085571);  convert_element_type_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_2: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, getitem_5);  view_6 = getitem_5 = None
        add_4: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_15: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_14: "f16[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg8_1, 0.05892556509887896);  arg8_1 = None
        view_7: "f16[64]" = torch.ops.aten.reshape.default(mul_14, [-1]);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_2: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(view_7, -1);  view_7 = None
        mul_16: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(mul_15, unsqueeze_2);  mul_15 = unsqueeze_2 = None
        convert_element_type_9: "f16[1, 64, 288]" = torch.ops.prims.convert_element_type.default(mul_16, torch.float16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_8: "f16[64, 32, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_9, [64, 32, 3, 3]);  convert_element_type_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_2: "f16[128, 64, 96, 96]" = torch.ops.aten.convolution.default(mul_13, view_8, arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_13 = view_8 = arg9_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_10: "f32[128, 64, 96, 96]" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        mul_17: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.5)
        mul_18: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.7071067811865476);  convert_element_type_10 = None
        erf_2: "f32[128, 64, 96, 96]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
        add_5: "f32[128, 64, 96, 96]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_19: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(mul_17, add_5);  mul_17 = add_5 = None
        convert_element_type_11: "f16[128, 64, 96, 96]" = torch.ops.prims.convert_element_type.default(mul_19, torch.float16);  mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_20: "f16[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.7015043497085571);  convert_element_type_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f16[128, 64, 97, 97]" = torch.ops.aten.constant_pad_nd.default(mul_20, [0, 1, 0, 1], 0.0);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_3: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, getitem_7);  view_9 = getitem_7 = None
        add_6: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_22: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_21: "f16[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg11_1, 0.041666666666666664);  arg11_1 = None
        view_10: "f16[128]" = torch.ops.aten.reshape.default(mul_21, [-1]);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_3: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(view_10, -1);  view_10 = None
        mul_23: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_3);  mul_22 = unsqueeze_3 = None
        convert_element_type_13: "f16[1, 128, 576]" = torch.ops.prims.convert_element_type.default(mul_23, torch.float16);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_11: "f16[128, 64, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_13, [128, 64, 3, 3]);  convert_element_type_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_3: "f16[128, 128, 48, 48]" = torch.ops.aten.convolution.default(constant_pad_nd_1, view_11, arg12_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  constant_pad_nd_1 = view_11 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_14: "f32[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        mul_24: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_25: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf_3: "f32[128, 128, 48, 48]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_7: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_26: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_24, add_7);  mul_24 = add_7 = None
        convert_element_type_15: "f16[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(mul_26, torch.float16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_27: "f16[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.7015043497085571);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_28: "f16[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_27, 1.0);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_5: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(view_15, getitem_11);  view_15 = getitem_11 = None
        add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_33: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_32: "f16[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg17_1, 0.08838834764831845);  arg17_1 = None
        view_16: "f16[128]" = torch.ops.aten.reshape.default(mul_32, [-1]);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_5: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(view_16, -1);  view_16 = None
        mul_34: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_5);  mul_33 = unsqueeze_5 = None
        convert_element_type_19: "f16[1, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_34, torch.float16);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_17: "f16[128, 128, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_19, [128, 128, 1, 1]);  convert_element_type_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_5: "f16[128, 128, 48, 48]" = torch.ops.aten.convolution.default(mul_28, view_17, arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_17 = arg18_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_20: "f32[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        mul_35: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 0.5)
        mul_36: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 0.7071067811865476);  convert_element_type_20 = None
        erf_4: "f32[128, 128, 48, 48]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_10: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_37: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_35, add_10);  mul_35 = add_10 = None
        convert_element_type_21: "f16[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(mul_37, torch.float16);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_38: "f16[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.7015043497085571);  convert_element_type_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_6: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_18, getitem_13);  view_18 = getitem_13 = None
        add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_40: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_39: "f16[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg20_1, 0.02946278254943948);  arg20_1 = None
        view_19: "f16[128]" = torch.ops.aten.reshape.default(mul_39, [-1]);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_6: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(view_19, -1);  view_19 = None
        mul_41: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_6);  mul_40 = unsqueeze_6 = None
        convert_element_type_23: "f16[1, 128, 1152]" = torch.ops.prims.convert_element_type.default(mul_41, torch.float16);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_20: "f16[128, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_23, [128, 128, 3, 3]);  convert_element_type_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_6: "f16[128, 128, 48, 48]" = torch.ops.aten.convolution.default(mul_38, view_20, arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_38 = view_20 = arg21_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_24: "f32[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        mul_42: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5)
        mul_43: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.7071067811865476);  convert_element_type_24 = None
        erf_5: "f32[128, 128, 48, 48]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_12: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_44: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_42, add_12);  mul_42 = add_12 = None
        convert_element_type_25: "f16[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(mul_44, torch.float16);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_45: "f16[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.7015043497085571);  convert_element_type_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_7: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_21, getitem_15);  view_21 = getitem_15 = None
        add_13: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_47: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_46: "f16[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg23_1, 0.02946278254943948);  arg23_1 = None
        view_22: "f16[128]" = torch.ops.aten.reshape.default(mul_46, [-1]);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_7: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(view_22, -1);  view_22 = None
        mul_48: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(mul_47, unsqueeze_7);  mul_47 = unsqueeze_7 = None
        convert_element_type_27: "f16[1, 128, 1152]" = torch.ops.prims.convert_element_type.default(mul_48, torch.float16);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_23: "f16[128, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_27, [128, 128, 3, 3]);  convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_7: "f16[128, 128, 48, 48]" = torch.ops.aten.convolution.default(mul_45, view_23, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_45 = view_23 = arg24_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_28: "f32[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        mul_49: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.5)
        mul_50: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.7071067811865476);  convert_element_type_28 = None
        erf_6: "f32[128, 128, 48, 48]" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_14: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_51: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_49, add_14);  mul_49 = add_14 = None
        convert_element_type_29: "f16[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(mul_51, torch.float16);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_52: "f16[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.7015043497085571);  convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_8: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_24, getitem_17);  view_24 = getitem_17 = None
        add_15: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_54: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_53: "f16[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg26_1, 0.08838834764831845);  arg26_1 = None
        view_25: "f16[256]" = torch.ops.aten.reshape.default(mul_53, [-1]);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_8: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(view_25, -1);  view_25 = None
        mul_55: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_54, unsqueeze_8);  mul_54 = unsqueeze_8 = None
        convert_element_type_31: "f16[1, 256, 128]" = torch.ops.prims.convert_element_type.default(mul_55, torch.float16);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_26: "f16[256, 128, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_31, [256, 128, 1, 1]);  convert_element_type_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_8: "f16[128, 256, 48, 48]" = torch.ops.aten.convolution.default(mul_52, view_26, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_52 = view_26 = arg27_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean: "f16[128, 256, 1, 1]" = torch.ops.aten.mean.dim(convolution_8, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_9: "f16[128, 128, 1, 1]" = torch.ops.aten.convolution.default(mean, arg28_1, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg28_1 = arg29_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu: "f16[128, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_10: "f16[128, 256, 1, 1]" = torch.ops.aten.convolution.default(relu, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu = arg30_1 = arg31_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid: "f16[128, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_56: "f16[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid);  convolution_8 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_57: "f16[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_56, 2.0);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_58: "f16[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_57, arg32_1);  mul_57 = arg32_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_59: "f16[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_58, 0.2);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_4: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, getitem_9);  view_12 = getitem_9 = None
        add_8: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_30: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_29: "f16[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg14_1, 0.08838834764831845);  arg14_1 = None
        view_13: "f16[256]" = torch.ops.aten.reshape.default(mul_29, [-1]);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_4: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(view_13, -1);  view_13 = None
        mul_31: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_30, unsqueeze_4);  mul_30 = unsqueeze_4 = None
        convert_element_type_17: "f16[1, 256, 128]" = torch.ops.prims.convert_element_type.default(mul_31, torch.float16);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_14: "f16[256, 128, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_17, [256, 128, 1, 1]);  convert_element_type_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_4: "f16[128, 256, 48, 48]" = torch.ops.aten.convolution.default(mul_28, view_14, arg15_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_28 = view_14 = arg15_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        add_16: "f16[128, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_59, convolution_4);  mul_59 = convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_32: "f32[128, 256, 48, 48]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        mul_60: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 0.5)
        mul_61: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 0.7071067811865476);  convert_element_type_32 = None
        erf_7: "f32[128, 256, 48, 48]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_17: "f32[128, 256, 48, 48]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_62: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_60, add_17);  mul_60 = add_17 = None
        convert_element_type_33: "f16[128, 256, 48, 48]" = torch.ops.prims.convert_element_type.default(mul_62, torch.float16);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_63: "f16[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.7015043497085571);  convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_64: "f16[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_63, 0.9805806756909201);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_10: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(view_30, getitem_21);  view_30 = getitem_21 = None
        add_19: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_69: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_68: "f16[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg37_1, 0.0625);  arg37_1 = None
        view_31: "f16[256]" = torch.ops.aten.reshape.default(mul_68, [-1]);  mul_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_10: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(view_31, -1);  view_31 = None
        mul_70: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(mul_69, unsqueeze_10);  mul_69 = unsqueeze_10 = None
        convert_element_type_37: "f16[1, 256, 256]" = torch.ops.prims.convert_element_type.default(mul_70, torch.float16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_32: "f16[256, 256, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_37, [256, 256, 1, 1]);  convert_element_type_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_12: "f16[128, 256, 48, 48]" = torch.ops.aten.convolution.default(mul_64, view_32, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_32 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_38: "f32[128, 256, 48, 48]" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        mul_71: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5)
        mul_72: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.7071067811865476);  convert_element_type_38 = None
        erf_8: "f32[128, 256, 48, 48]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
        add_20: "f32[128, 256, 48, 48]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_73: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_71, add_20);  mul_71 = add_20 = None
        convert_element_type_39: "f16[128, 256, 48, 48]" = torch.ops.prims.convert_element_type.default(mul_73, torch.float16);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_74: "f16[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.7015043497085571);  convert_element_type_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "f16[128, 256, 49, 49]" = torch.ops.aten.constant_pad_nd.default(mul_74, [0, 1, 0, 1], 0.0);  mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_11: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_33, getitem_23);  view_33 = getitem_23 = None
        add_21: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_76: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_75: "f16[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg40_1, 0.02946278254943948);  arg40_1 = None
        view_34: "f16[256]" = torch.ops.aten.reshape.default(mul_75, [-1]);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_11: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(view_34, -1);  view_34 = None
        mul_77: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_11);  mul_76 = unsqueeze_11 = None
        convert_element_type_41: "f16[1, 256, 1152]" = torch.ops.prims.convert_element_type.default(mul_77, torch.float16);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_35: "f16[256, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_41, [256, 128, 3, 3]);  convert_element_type_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_13: "f16[128, 256, 24, 24]" = torch.ops.aten.convolution.default(constant_pad_nd_2, view_35, arg41_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 2);  constant_pad_nd_2 = view_35 = arg41_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_42: "f32[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        mul_78: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.5)
        mul_79: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.7071067811865476);  convert_element_type_42 = None
        erf_9: "f32[128, 256, 24, 24]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_22: "f32[128, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_78, add_22);  mul_78 = add_22 = None
        convert_element_type_43: "f16[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_80, torch.float16);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_81: "f16[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.7015043497085571);  convert_element_type_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_12: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_36, getitem_25);  view_36 = getitem_25 = None
        add_23: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_83: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_82: "f16[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg43_1, 0.02946278254943948);  arg43_1 = None
        view_37: "f16[256]" = torch.ops.aten.reshape.default(mul_82, [-1]);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_12: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(view_37, -1);  view_37 = None
        mul_84: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_12);  mul_83 = unsqueeze_12 = None
        convert_element_type_45: "f16[1, 256, 1152]" = torch.ops.prims.convert_element_type.default(mul_84, torch.float16);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_38: "f16[256, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_45, [256, 128, 3, 3]);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_14: "f16[128, 256, 24, 24]" = torch.ops.aten.convolution.default(mul_81, view_38, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_81 = view_38 = arg44_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_46: "f32[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        mul_85: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.5)
        mul_86: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.7071067811865476);  convert_element_type_46 = None
        erf_10: "f32[128, 256, 24, 24]" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_24: "f32[128, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_87: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_85, add_24);  mul_85 = add_24 = None
        convert_element_type_47: "f16[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_87, torch.float16);  mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_88: "f16[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.7015043497085571);  convert_element_type_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_13: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_39, getitem_27);  view_39 = getitem_27 = None
        add_25: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_90: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_89: "f16[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg46_1, 0.0625);  arg46_1 = None
        view_40: "f16[512]" = torch.ops.aten.reshape.default(mul_89, [-1]);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_13: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(view_40, -1);  view_40 = None
        mul_91: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_90, unsqueeze_13);  mul_90 = unsqueeze_13 = None
        convert_element_type_49: "f16[1, 512, 256]" = torch.ops.prims.convert_element_type.default(mul_91, torch.float16);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_41: "f16[512, 256, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_49, [512, 256, 1, 1]);  convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_15: "f16[128, 512, 24, 24]" = torch.ops.aten.convolution.default(mul_88, view_41, arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_88 = view_41 = arg47_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_1: "f16[128, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_15, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_16: "f16[128, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg48_1, arg49_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg48_1 = arg49_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_1: "f16[128, 256, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_17: "f16[128, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_1, arg50_1, arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_1 = arg50_1 = arg51_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_1: "f16[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_92: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_1);  convolution_15 = sigmoid_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_93: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_92, 2.0);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_94: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_93, arg52_1);  mul_93 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_95: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_94, 0.2);  mul_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d: "f16[128, 256, 24, 24]" = torch.ops.aten.avg_pool2d.default(mul_64, [2, 2], [2, 2], [0, 0], True, False);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, getitem_19);  view_27 = getitem_19 = None
        add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_66: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_65: "f16[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg34_1, 0.0625);  arg34_1 = None
        view_28: "f16[512]" = torch.ops.aten.reshape.default(mul_65, [-1]);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_9: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(view_28, -1);  view_28 = None
        mul_67: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_9);  mul_66 = unsqueeze_9 = None
        convert_element_type_35: "f16[1, 512, 256]" = torch.ops.prims.convert_element_type.default(mul_67, torch.float16);  mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_29: "f16[512, 256, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_35, [512, 256, 1, 1]);  convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_11: "f16[128, 512, 24, 24]" = torch.ops.aten.convolution.default(avg_pool2d, view_29, arg35_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d = view_29 = arg35_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        add_26: "f16[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_95, convolution_11);  mul_95 = convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_50: "f32[128, 512, 24, 24]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)
        mul_96: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 0.5)
        mul_97: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 0.7071067811865476);  convert_element_type_50 = None
        erf_11: "f32[128, 512, 24, 24]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_27: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_98: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_96, add_27);  mul_96 = add_27 = None
        convert_element_type_51: "f16[128, 512, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_98, torch.float16);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_99: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.7015043497085571);  convert_element_type_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_100: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_99, 0.9805806756909201);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_14: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(view_42, getitem_29);  view_42 = getitem_29 = None
        add_28: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_102: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_101: "f16[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg54_1, 0.04419417382415922);  arg54_1 = None
        view_43: "f16[256]" = torch.ops.aten.reshape.default(mul_101, [-1]);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_14: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(view_43, -1);  view_43 = None
        mul_103: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_14);  mul_102 = unsqueeze_14 = None
        convert_element_type_53: "f16[1, 256, 512]" = torch.ops.prims.convert_element_type.default(mul_103, torch.float16);  mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_44: "f16[256, 512, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_53, [256, 512, 1, 1]);  convert_element_type_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_18: "f16[128, 256, 24, 24]" = torch.ops.aten.convolution.default(mul_100, view_44, arg55_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_100 = view_44 = arg55_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_54: "f32[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        mul_104: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.5)
        mul_105: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.7071067811865476);  convert_element_type_54 = None
        erf_12: "f32[128, 256, 24, 24]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_29: "f32[128, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_106: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_104, add_29);  mul_104 = add_29 = None
        convert_element_type_55: "f16[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_106, torch.float16);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_107: "f16[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.7015043497085571);  convert_element_type_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_15: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_45, getitem_31);  view_45 = getitem_31 = None
        add_30: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_109: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_108: "f16[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg57_1, 0.02946278254943948);  arg57_1 = None
        view_46: "f16[256]" = torch.ops.aten.reshape.default(mul_108, [-1]);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_15: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(view_46, -1);  view_46 = None
        mul_110: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_15);  mul_109 = unsqueeze_15 = None
        convert_element_type_57: "f16[1, 256, 1152]" = torch.ops.prims.convert_element_type.default(mul_110, torch.float16);  mul_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_47: "f16[256, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_57, [256, 128, 3, 3]);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_19: "f16[128, 256, 24, 24]" = torch.ops.aten.convolution.default(mul_107, view_47, arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_107 = view_47 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_58: "f32[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        mul_111: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.5)
        mul_112: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.7071067811865476);  convert_element_type_58 = None
        erf_13: "f32[128, 256, 24, 24]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_31: "f32[128, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_113: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_111, add_31);  mul_111 = add_31 = None
        convert_element_type_59: "f16[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_113, torch.float16);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_114: "f16[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.7015043497085571);  convert_element_type_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_16: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_48, getitem_33);  view_48 = getitem_33 = None
        add_32: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_116: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_115: "f16[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg60_1, 0.02946278254943948);  arg60_1 = None
        view_49: "f16[256]" = torch.ops.aten.reshape.default(mul_115, [-1]);  mul_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_16: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(view_49, -1);  view_49 = None
        mul_117: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_16);  mul_116 = unsqueeze_16 = None
        convert_element_type_61: "f16[1, 256, 1152]" = torch.ops.prims.convert_element_type.default(mul_117, torch.float16);  mul_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_50: "f16[256, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_61, [256, 128, 3, 3]);  convert_element_type_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_20: "f16[128, 256, 24, 24]" = torch.ops.aten.convolution.default(mul_114, view_50, arg61_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_114 = view_50 = arg61_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_62: "f32[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        mul_118: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.5)
        mul_119: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.7071067811865476);  convert_element_type_62 = None
        erf_14: "f32[128, 256, 24, 24]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_33: "f32[128, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_120: "f32[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_118, add_33);  mul_118 = add_33 = None
        convert_element_type_63: "f16[128, 256, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_120, torch.float16);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_121: "f16[128, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.7015043497085571);  convert_element_type_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_17: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_51, getitem_35);  view_51 = getitem_35 = None
        add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_123: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_122: "f16[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg63_1, 0.0625);  arg63_1 = None
        view_52: "f16[512]" = torch.ops.aten.reshape.default(mul_122, [-1]);  mul_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_17: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(view_52, -1);  view_52 = None
        mul_124: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_17);  mul_123 = unsqueeze_17 = None
        convert_element_type_65: "f16[1, 512, 256]" = torch.ops.prims.convert_element_type.default(mul_124, torch.float16);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_53: "f16[512, 256, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_65, [512, 256, 1, 1]);  convert_element_type_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_21: "f16[128, 512, 24, 24]" = torch.ops.aten.convolution.default(mul_121, view_53, arg64_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_121 = view_53 = arg64_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_2: "f16[128, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_21, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_22: "f16[128, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_2, arg65_1, arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg65_1 = arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_2: "f16[128, 256, 1, 1]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_23: "f16[128, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_2, arg67_1, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg67_1 = arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_2: "f16[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_125: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_2);  convolution_21 = sigmoid_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_126: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_125, 2.0);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_127: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_126, arg69_1);  mul_126 = arg69_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_128: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_127, 0.2);  mul_127 = None
        add_35: "f16[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_128, add_26);  mul_128 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_66: "f32[128, 512, 24, 24]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        mul_129: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 0.5)
        mul_130: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 0.7071067811865476);  convert_element_type_66 = None
        erf_15: "f32[128, 512, 24, 24]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
        add_36: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_131: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_129, add_36);  mul_129 = add_36 = None
        convert_element_type_67: "f16[128, 512, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_131, torch.float16);  mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_132: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.7015043497085571);  convert_element_type_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_133: "f16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_132, 0.9622504486493761);  mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_19: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(view_57, getitem_39);  view_57 = getitem_39 = None
        add_38: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_138: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_137: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg74_1, 0.04419417382415922);  arg74_1 = None
        view_58: "f16[768]" = torch.ops.aten.reshape.default(mul_137, [-1]);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_19: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_58, -1);  view_58 = None
        mul_139: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_19);  mul_138 = unsqueeze_19 = None
        convert_element_type_71: "f16[1, 768, 512]" = torch.ops.prims.convert_element_type.default(mul_139, torch.float16);  mul_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_59: "f16[768, 512, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_71, [768, 512, 1, 1]);  convert_element_type_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_25: "f16[128, 768, 24, 24]" = torch.ops.aten.convolution.default(mul_133, view_59, arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_59 = arg75_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_72: "f32[128, 768, 24, 24]" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        mul_140: "f32[128, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 0.5)
        mul_141: "f32[128, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 0.7071067811865476);  convert_element_type_72 = None
        erf_16: "f32[128, 768, 24, 24]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_39: "f32[128, 768, 24, 24]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_142: "f32[128, 768, 24, 24]" = torch.ops.aten.mul.Tensor(mul_140, add_39);  mul_140 = add_39 = None
        convert_element_type_73: "f16[128, 768, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_142, torch.float16);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_143: "f16[128, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.7015043497085571);  convert_element_type_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "f16[128, 768, 25, 25]" = torch.ops.aten.constant_pad_nd.default(mul_143, [0, 1, 0, 1], 0.0);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_20: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_60, getitem_41);  view_60 = getitem_41 = None
        add_40: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_145: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_144: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg77_1, 0.02946278254943948);  arg77_1 = None
        view_61: "f16[768]" = torch.ops.aten.reshape.default(mul_144, [-1]);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_20: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_61, -1);  view_61 = None
        mul_146: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_20);  mul_145 = unsqueeze_20 = None
        convert_element_type_75: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_146, torch.float16);  mul_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_62: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_75, [768, 128, 3, 3]);  convert_element_type_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_26: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(constant_pad_nd_3, view_62, arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6);  constant_pad_nd_3 = view_62 = arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_76: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        mul_147: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 0.5)
        mul_148: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 0.7071067811865476);  convert_element_type_76 = None
        erf_17: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_41: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_149: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_147, add_41);  mul_147 = add_41 = None
        convert_element_type_77: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_149, torch.float16);  mul_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_150: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_77, 1.7015043497085571);  convert_element_type_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_21: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_63, getitem_43);  view_63 = getitem_43 = None
        add_42: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_152: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_151: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg80_1, 0.02946278254943948);  arg80_1 = None
        view_64: "f16[768]" = torch.ops.aten.reshape.default(mul_151, [-1]);  mul_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_21: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_64, -1);  view_64 = None
        mul_153: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_152, unsqueeze_21);  mul_152 = unsqueeze_21 = None
        convert_element_type_79: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_153, torch.float16);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_65: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_79, [768, 128, 3, 3]);  convert_element_type_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_27: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_150, view_65, arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_150 = view_65 = arg81_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_80: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        mul_154: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 0.5)
        mul_155: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 0.7071067811865476);  convert_element_type_80 = None
        erf_18: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
        add_43: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_156: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_154, add_43);  mul_154 = add_43 = None
        convert_element_type_81: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_156, torch.float16);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_157: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 1.7015043497085571);  convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_22: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_66, getitem_45);  view_66 = getitem_45 = None
        add_44: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_159: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_158: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg83_1, 0.03608439182435161);  arg83_1 = None
        view_67: "f16[1536]" = torch.ops.aten.reshape.default(mul_158, [-1]);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_22: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_67, -1);  view_67 = None
        mul_160: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_22);  mul_159 = unsqueeze_22 = None
        convert_element_type_83: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_160, torch.float16);  mul_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_68: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_83, [1536, 768, 1, 1]);  convert_element_type_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_28: "f16[128, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_157, view_68, arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_157 = view_68 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_3: "f16[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_28, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_29: "f16[128, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_3, arg85_1, arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg85_1 = arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_3: "f16[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_29);  convolution_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_30: "f16[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_3, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_3 = arg87_1 = arg88_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_3: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_161: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_3);  convolution_28 = sigmoid_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_162: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_161, 2.0);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_163: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_162, arg89_1);  mul_162 = arg89_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_164: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_163, 0.2);  mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_1: "f16[128, 512, 12, 12]" = torch.ops.aten.avg_pool2d.default(mul_133, [2, 2], [2, 2], [0, 0], True, False);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_18: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, getitem_37);  view_54 = getitem_37 = None
        add_37: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_135: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_134: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg71_1, 0.04419417382415922);  arg71_1 = None
        view_55: "f16[1536]" = torch.ops.aten.reshape.default(mul_134, [-1]);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_18: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_55, -1);  view_55 = None
        mul_136: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_18);  mul_135 = unsqueeze_18 = None
        convert_element_type_69: "f16[1, 1536, 512]" = torch.ops.prims.convert_element_type.default(mul_136, torch.float16);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_56: "f16[1536, 512, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_69, [1536, 512, 1, 1]);  convert_element_type_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_24: "f16[128, 1536, 12, 12]" = torch.ops.aten.convolution.default(avg_pool2d_1, view_56, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_1 = view_56 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        add_45: "f16[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_164, convolution_24);  mul_164 = convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_84: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        mul_165: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 0.5)
        mul_166: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 0.7071067811865476);  convert_element_type_84 = None
        erf_19: "f32[128, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
        add_46: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_167: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_165, add_46);  mul_165 = add_46 = None
        convert_element_type_85: "f16[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_167, torch.float16);  mul_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_168: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 1.7015043497085571);  convert_element_type_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_169: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_168, 0.9805806756909201);  mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_23: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_69, getitem_47);  view_69 = getitem_47 = None
        add_47: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_171: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_170: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg91_1, 0.02551551815399144);  arg91_1 = None
        view_70: "f16[768]" = torch.ops.aten.reshape.default(mul_170, [-1]);  mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_23: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_70, -1);  view_70 = None
        mul_172: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_23);  mul_171 = unsqueeze_23 = None
        convert_element_type_87: "f16[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(mul_172, torch.float16);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_71: "f16[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_87, [768, 1536, 1, 1]);  convert_element_type_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_31: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_169, view_71, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_169 = view_71 = arg92_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_88: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        mul_173: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.5)
        mul_174: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.7071067811865476);  convert_element_type_88 = None
        erf_20: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_48: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_175: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_173, add_48);  mul_173 = add_48 = None
        convert_element_type_89: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_175, torch.float16);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_176: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.7015043497085571);  convert_element_type_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_24: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_72, getitem_49);  view_72 = getitem_49 = None
        add_49: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_178: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_177: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg94_1, 0.02946278254943948);  arg94_1 = None
        view_73: "f16[768]" = torch.ops.aten.reshape.default(mul_177, [-1]);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_24: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_73, -1);  view_73 = None
        mul_179: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_24);  mul_178 = unsqueeze_24 = None
        convert_element_type_91: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_179, torch.float16);  mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_74: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_91, [768, 128, 3, 3]);  convert_element_type_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_32: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_176, view_74, arg95_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_176 = view_74 = arg95_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_92: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        mul_180: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.5)
        mul_181: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.7071067811865476);  convert_element_type_92 = None
        erf_21: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_50: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_182: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_180, add_50);  mul_180 = add_50 = None
        convert_element_type_93: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_182, torch.float16);  mul_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_183: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_93, 1.7015043497085571);  convert_element_type_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_25: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_75, getitem_51);  view_75 = getitem_51 = None
        add_51: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_185: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_184: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg97_1, 0.02946278254943948);  arg97_1 = None
        view_76: "f16[768]" = torch.ops.aten.reshape.default(mul_184, [-1]);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_25: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_76, -1);  view_76 = None
        mul_186: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_185, unsqueeze_25);  mul_185 = unsqueeze_25 = None
        convert_element_type_95: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_186, torch.float16);  mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_77: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_95, [768, 128, 3, 3]);  convert_element_type_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_33: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_183, view_77, arg98_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_183 = view_77 = arg98_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_96: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        mul_187: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_96, 0.5)
        mul_188: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_96, 0.7071067811865476);  convert_element_type_96 = None
        erf_22: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
        add_52: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_189: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_187, add_52);  mul_187 = add_52 = None
        convert_element_type_97: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_189, torch.float16);  mul_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_190: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 1.7015043497085571);  convert_element_type_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_26: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_78, getitem_53);  view_78 = getitem_53 = None
        add_53: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_192: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_191: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg100_1, 0.03608439182435161);  arg100_1 = None
        view_79: "f16[1536]" = torch.ops.aten.reshape.default(mul_191, [-1]);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_26: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_79, -1);  view_79 = None
        mul_193: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_26);  mul_192 = unsqueeze_26 = None
        convert_element_type_99: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_193, torch.float16);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_80: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_99, [1536, 768, 1, 1]);  convert_element_type_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_34: "f16[128, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_190, view_80, arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_190 = view_80 = arg101_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_4: "f16[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_34, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_35: "f16[128, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_4, arg102_1, arg103_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg102_1 = arg103_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_4: "f16[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_36: "f16[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_4, arg104_1, arg105_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg104_1 = arg105_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_4: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_194: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_34, sigmoid_4);  convolution_34 = sigmoid_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_195: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_194, 2.0);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_196: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_195, arg106_1);  mul_195 = arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_197: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_196, 0.2);  mul_196 = None
        add_54: "f16[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_197, add_45);  mul_197 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_100: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(add_54, torch.float32)
        mul_198: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 0.5)
        mul_199: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 0.7071067811865476);  convert_element_type_100 = None
        erf_23: "f32[128, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_199);  mul_199 = None
        add_55: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_200: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_198, add_55);  mul_198 = add_55 = None
        convert_element_type_101: "f16[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_200, torch.float16);  mul_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_201: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_101, 1.7015043497085571);  convert_element_type_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_202: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_201, 0.9622504486493761);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_27: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_81, getitem_55);  view_81 = getitem_55 = None
        add_56: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_204: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_203: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg108_1, 0.02551551815399144);  arg108_1 = None
        view_82: "f16[768]" = torch.ops.aten.reshape.default(mul_203, [-1]);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_27: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_82, -1);  view_82 = None
        mul_205: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_27);  mul_204 = unsqueeze_27 = None
        convert_element_type_103: "f16[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(mul_205, torch.float16);  mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_83: "f16[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_103, [768, 1536, 1, 1]);  convert_element_type_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_37: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_202, view_83, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_202 = view_83 = arg109_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_104: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        mul_206: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_104, 0.5)
        mul_207: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_104, 0.7071067811865476);  convert_element_type_104 = None
        erf_24: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_57: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_208: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_206, add_57);  mul_206 = add_57 = None
        convert_element_type_105: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_208, torch.float16);  mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_209: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_105, 1.7015043497085571);  convert_element_type_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_28: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_84, getitem_57);  view_84 = getitem_57 = None
        add_58: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_211: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_210: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg111_1, 0.02946278254943948);  arg111_1 = None
        view_85: "f16[768]" = torch.ops.aten.reshape.default(mul_210, [-1]);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_28: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_85, -1);  view_85 = None
        mul_212: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_28);  mul_211 = unsqueeze_28 = None
        convert_element_type_107: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_212, torch.float16);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_86: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_107, [768, 128, 3, 3]);  convert_element_type_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_38: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_209, view_86, arg112_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_209 = view_86 = arg112_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_108: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32);  convolution_38 = None
        mul_213: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_108, 0.5)
        mul_214: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_108, 0.7071067811865476);  convert_element_type_108 = None
        erf_25: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_214);  mul_214 = None
        add_59: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_215: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_213, add_59);  mul_213 = add_59 = None
        convert_element_type_109: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_215, torch.float16);  mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_216: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_109, 1.7015043497085571);  convert_element_type_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_29: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_87, getitem_59);  view_87 = getitem_59 = None
        add_60: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_218: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_217: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg114_1, 0.02946278254943948);  arg114_1 = None
        view_88: "f16[768]" = torch.ops.aten.reshape.default(mul_217, [-1]);  mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_29: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_88, -1);  view_88 = None
        mul_219: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_29);  mul_218 = unsqueeze_29 = None
        convert_element_type_111: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_219, torch.float16);  mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_89: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_111, [768, 128, 3, 3]);  convert_element_type_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_39: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_216, view_89, arg115_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_216 = view_89 = arg115_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_112: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        mul_220: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_112, 0.5)
        mul_221: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_112, 0.7071067811865476);  convert_element_type_112 = None
        erf_26: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
        add_61: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_222: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_220, add_61);  mul_220 = add_61 = None
        convert_element_type_113: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_222, torch.float16);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_223: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_113, 1.7015043497085571);  convert_element_type_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_30: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_90, getitem_61);  view_90 = getitem_61 = None
        add_62: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_225: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_224: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg117_1, 0.03608439182435161);  arg117_1 = None
        view_91: "f16[1536]" = torch.ops.aten.reshape.default(mul_224, [-1]);  mul_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_30: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_91, -1);  view_91 = None
        mul_226: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_30);  mul_225 = unsqueeze_30 = None
        convert_element_type_115: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_226, torch.float16);  mul_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_92: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_115, [1536, 768, 1, 1]);  convert_element_type_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_40: "f16[128, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_223, view_92, arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_223 = view_92 = arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_5: "f16[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_40, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_41: "f16[128, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_5, arg119_1, arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg119_1 = arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_5: "f16[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_41);  convolution_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_42: "f16[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_5, arg121_1, arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg121_1 = arg122_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_5: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_227: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_40, sigmoid_5);  convolution_40 = sigmoid_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_228: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_227, 2.0);  mul_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_229: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_228, arg123_1);  mul_228 = arg123_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_230: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_229, 0.2);  mul_229 = None
        add_63: "f16[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_230, add_54);  mul_230 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_116: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        mul_231: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.5)
        mul_232: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.7071067811865476);  convert_element_type_116 = None
        erf_27: "f32[128, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_232);  mul_232 = None
        add_64: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_233: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_231, add_64);  mul_231 = add_64 = None
        convert_element_type_117: "f16[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_233, torch.float16);  mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_234: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_117, 1.7015043497085571);  convert_element_type_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_235: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_234, 0.9449111825230679);  mul_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_31: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_93, getitem_63);  view_93 = getitem_63 = None
        add_65: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_237: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_236: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg125_1, 0.02551551815399144);  arg125_1 = None
        view_94: "f16[768]" = torch.ops.aten.reshape.default(mul_236, [-1]);  mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_31: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_94, -1);  view_94 = None
        mul_238: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_31);  mul_237 = unsqueeze_31 = None
        convert_element_type_119: "f16[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(mul_238, torch.float16);  mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_95: "f16[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_119, [768, 1536, 1, 1]);  convert_element_type_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_43: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_235, view_95, arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_235 = view_95 = arg126_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_120: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        mul_239: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_120, 0.5)
        mul_240: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_120, 0.7071067811865476);  convert_element_type_120 = None
        erf_28: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
        add_66: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_241: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_239, add_66);  mul_239 = add_66 = None
        convert_element_type_121: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_241, torch.float16);  mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_242: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_121, 1.7015043497085571);  convert_element_type_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_32: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_96, getitem_65);  view_96 = getitem_65 = None
        add_67: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_244: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_243: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg128_1, 0.02946278254943948);  arg128_1 = None
        view_97: "f16[768]" = torch.ops.aten.reshape.default(mul_243, [-1]);  mul_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_32: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_97, -1);  view_97 = None
        mul_245: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_32);  mul_244 = unsqueeze_32 = None
        convert_element_type_123: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_245, torch.float16);  mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_98: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_123, [768, 128, 3, 3]);  convert_element_type_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_44: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_242, view_98, arg129_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_242 = view_98 = arg129_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_124: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        mul_246: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 0.5)
        mul_247: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 0.7071067811865476);  convert_element_type_124 = None
        erf_29: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_247);  mul_247 = None
        add_68: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_248: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_246, add_68);  mul_246 = add_68 = None
        convert_element_type_125: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_248, torch.float16);  mul_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_249: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_125, 1.7015043497085571);  convert_element_type_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_33: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_99, getitem_67);  view_99 = getitem_67 = None
        add_69: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_251: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_250: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg131_1, 0.02946278254943948);  arg131_1 = None
        view_100: "f16[768]" = torch.ops.aten.reshape.default(mul_250, [-1]);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_33: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_100, -1);  view_100 = None
        mul_252: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_33);  mul_251 = unsqueeze_33 = None
        convert_element_type_127: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_252, torch.float16);  mul_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_101: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_127, [768, 128, 3, 3]);  convert_element_type_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_45: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_249, view_101, arg132_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_249 = view_101 = arg132_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_128: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        mul_253: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_128, 0.5)
        mul_254: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_128, 0.7071067811865476);  convert_element_type_128 = None
        erf_30: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_254);  mul_254 = None
        add_70: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_255: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_253, add_70);  mul_253 = add_70 = None
        convert_element_type_129: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_255, torch.float16);  mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_256: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_129, 1.7015043497085571);  convert_element_type_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_34: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_102, getitem_69);  view_102 = getitem_69 = None
        add_71: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_258: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_257: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg134_1, 0.03608439182435161);  arg134_1 = None
        view_103: "f16[1536]" = torch.ops.aten.reshape.default(mul_257, [-1]);  mul_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_34: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_103, -1);  view_103 = None
        mul_259: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_34);  mul_258 = unsqueeze_34 = None
        convert_element_type_131: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_259, torch.float16);  mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_104: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_131, [1536, 768, 1, 1]);  convert_element_type_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_46: "f16[128, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_256, view_104, arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_256 = view_104 = arg135_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_6: "f16[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_46, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_47: "f16[128, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_6, arg136_1, arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg136_1 = arg137_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_6: "f16[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_48: "f16[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_6, arg138_1, arg139_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_6 = arg138_1 = arg139_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_6: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_260: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_46, sigmoid_6);  convolution_46 = sigmoid_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_261: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_260, 2.0);  mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_262: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_261, arg140_1);  mul_261 = arg140_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_263: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_262, 0.2);  mul_262 = None
        add_72: "f16[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_263, add_63);  mul_263 = add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_132: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(add_72, torch.float32)
        mul_264: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_132, 0.5)
        mul_265: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_132, 0.7071067811865476);  convert_element_type_132 = None
        erf_31: "f32[128, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_265);  mul_265 = None
        add_73: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_266: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_264, add_73);  mul_264 = add_73 = None
        convert_element_type_133: "f16[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_266, torch.float16);  mul_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_267: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_133, 1.7015043497085571);  convert_element_type_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_268: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_267, 0.9284766908852592);  mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_35: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_105, getitem_71);  view_105 = getitem_71 = None
        add_74: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_270: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_269: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg142_1, 0.02551551815399144);  arg142_1 = None
        view_106: "f16[768]" = torch.ops.aten.reshape.default(mul_269, [-1]);  mul_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_35: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_106, -1);  view_106 = None
        mul_271: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_35);  mul_270 = unsqueeze_35 = None
        convert_element_type_135: "f16[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(mul_271, torch.float16);  mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_107: "f16[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_135, [768, 1536, 1, 1]);  convert_element_type_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_49: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_268, view_107, arg143_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_268 = view_107 = arg143_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_136: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        mul_272: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.5)
        mul_273: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.7071067811865476);  convert_element_type_136 = None
        erf_32: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_273);  mul_273 = None
        add_75: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_274: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_272, add_75);  mul_272 = add_75 = None
        convert_element_type_137: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_274, torch.float16);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_275: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_137, 1.7015043497085571);  convert_element_type_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_36: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_108, getitem_73);  view_108 = getitem_73 = None
        add_76: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_277: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_276: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg145_1, 0.02946278254943948);  arg145_1 = None
        view_109: "f16[768]" = torch.ops.aten.reshape.default(mul_276, [-1]);  mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_36: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_109, -1);  view_109 = None
        mul_278: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_36);  mul_277 = unsqueeze_36 = None
        convert_element_type_139: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_278, torch.float16);  mul_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_110: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_139, [768, 128, 3, 3]);  convert_element_type_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_50: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_275, view_110, arg146_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_275 = view_110 = arg146_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_140: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        mul_279: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.5)
        mul_280: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.7071067811865476);  convert_element_type_140 = None
        erf_33: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_280);  mul_280 = None
        add_77: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_281: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_279, add_77);  mul_279 = add_77 = None
        convert_element_type_141: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_281, torch.float16);  mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_282: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_141, 1.7015043497085571);  convert_element_type_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_37: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_111, getitem_75);  view_111 = getitem_75 = None
        add_78: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_284: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_283: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg148_1, 0.02946278254943948);  arg148_1 = None
        view_112: "f16[768]" = torch.ops.aten.reshape.default(mul_283, [-1]);  mul_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_37: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_112, -1);  view_112 = None
        mul_285: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_37);  mul_284 = unsqueeze_37 = None
        convert_element_type_143: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_285, torch.float16);  mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_113: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_143, [768, 128, 3, 3]);  convert_element_type_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_51: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_282, view_113, arg149_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_282 = view_113 = arg149_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_144: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        mul_286: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_144, 0.5)
        mul_287: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_144, 0.7071067811865476);  convert_element_type_144 = None
        erf_34: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_287);  mul_287 = None
        add_79: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_288: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_286, add_79);  mul_286 = add_79 = None
        convert_element_type_145: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_288, torch.float16);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_289: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_145, 1.7015043497085571);  convert_element_type_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_38: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_114, getitem_77);  view_114 = getitem_77 = None
        add_80: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_291: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_290: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg151_1, 0.03608439182435161);  arg151_1 = None
        view_115: "f16[1536]" = torch.ops.aten.reshape.default(mul_290, [-1]);  mul_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_38: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_115, -1);  view_115 = None
        mul_292: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_291, unsqueeze_38);  mul_291 = unsqueeze_38 = None
        convert_element_type_147: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_292, torch.float16);  mul_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_116: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_147, [1536, 768, 1, 1]);  convert_element_type_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_52: "f16[128, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_289, view_116, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_289 = view_116 = arg152_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_7: "f16[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_52, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_53: "f16[128, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_7, arg153_1, arg154_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg153_1 = arg154_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_7: "f16[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_54: "f16[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_7, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_7 = arg155_1 = arg156_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_7: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_54);  convolution_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_293: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_7);  convolution_52 = sigmoid_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_294: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_293, 2.0);  mul_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_295: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_294, arg157_1);  mul_294 = arg157_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_296: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_295, 0.2);  mul_295 = None
        add_81: "f16[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_296, add_72);  mul_296 = add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_148: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        mul_297: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.5)
        mul_298: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.7071067811865476);  convert_element_type_148 = None
        erf_35: "f32[128, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_298);  mul_298 = None
        add_82: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_299: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_297, add_82);  mul_297 = add_82 = None
        convert_element_type_149: "f16[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_299, torch.float16);  mul_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_300: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_149, 1.7015043497085571);  convert_element_type_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_301: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_300, 0.9128709291752768);  mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_39: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_117, getitem_79);  view_117 = getitem_79 = None
        add_83: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_303: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_302: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg159_1, 0.02551551815399144);  arg159_1 = None
        view_118: "f16[768]" = torch.ops.aten.reshape.default(mul_302, [-1]);  mul_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_39: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_118, -1);  view_118 = None
        mul_304: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_303, unsqueeze_39);  mul_303 = unsqueeze_39 = None
        convert_element_type_151: "f16[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(mul_304, torch.float16);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_119: "f16[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_151, [768, 1536, 1, 1]);  convert_element_type_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_55: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_301, view_119, arg160_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_301 = view_119 = arg160_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_152: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        mul_305: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_152, 0.5)
        mul_306: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_152, 0.7071067811865476);  convert_element_type_152 = None
        erf_36: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_306);  mul_306 = None
        add_84: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_307: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_305, add_84);  mul_305 = add_84 = None
        convert_element_type_153: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_307, torch.float16);  mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_308: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_153, 1.7015043497085571);  convert_element_type_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_40: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_120, getitem_81);  view_120 = getitem_81 = None
        add_85: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_310: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_309: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg162_1, 0.02946278254943948);  arg162_1 = None
        view_121: "f16[768]" = torch.ops.aten.reshape.default(mul_309, [-1]);  mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_40: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_121, -1);  view_121 = None
        mul_311: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_40);  mul_310 = unsqueeze_40 = None
        convert_element_type_155: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_311, torch.float16);  mul_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_122: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_155, [768, 128, 3, 3]);  convert_element_type_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_56: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_308, view_122, arg163_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_308 = view_122 = arg163_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_156: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32);  convolution_56 = None
        mul_312: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_156, 0.5)
        mul_313: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_156, 0.7071067811865476);  convert_element_type_156 = None
        erf_37: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_313);  mul_313 = None
        add_86: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_314: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_312, add_86);  mul_312 = add_86 = None
        convert_element_type_157: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_314, torch.float16);  mul_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_315: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_157, 1.7015043497085571);  convert_element_type_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_41: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_123, getitem_83);  view_123 = getitem_83 = None
        add_87: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_317: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_316: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg165_1, 0.02946278254943948);  arg165_1 = None
        view_124: "f16[768]" = torch.ops.aten.reshape.default(mul_316, [-1]);  mul_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_41: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_124, -1);  view_124 = None
        mul_318: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_41);  mul_317 = unsqueeze_41 = None
        convert_element_type_159: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_318, torch.float16);  mul_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_125: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_159, [768, 128, 3, 3]);  convert_element_type_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_57: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_315, view_125, arg166_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_315 = view_125 = arg166_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_160: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32);  convolution_57 = None
        mul_319: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_160, 0.5)
        mul_320: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_160, 0.7071067811865476);  convert_element_type_160 = None
        erf_38: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_320);  mul_320 = None
        add_88: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_321: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_319, add_88);  mul_319 = add_88 = None
        convert_element_type_161: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_321, torch.float16);  mul_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_322: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_161, 1.7015043497085571);  convert_element_type_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_42: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_126, getitem_85);  view_126 = getitem_85 = None
        add_89: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_324: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_323: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg168_1, 0.03608439182435161);  arg168_1 = None
        view_127: "f16[1536]" = torch.ops.aten.reshape.default(mul_323, [-1]);  mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_42: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_127, -1);  view_127 = None
        mul_325: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_324, unsqueeze_42);  mul_324 = unsqueeze_42 = None
        convert_element_type_163: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_325, torch.float16);  mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_128: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_163, [1536, 768, 1, 1]);  convert_element_type_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_58: "f16[128, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_322, view_128, arg169_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_322 = view_128 = arg169_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_8: "f16[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_58, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_59: "f16[128, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_8, arg170_1, arg171_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg170_1 = arg171_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_8: "f16[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_59);  convolution_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_60: "f16[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_8, arg172_1, arg173_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg172_1 = arg173_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_8: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_326: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_8);  convolution_58 = sigmoid_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_327: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_326, 2.0);  mul_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_328: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_327, arg174_1);  mul_327 = arg174_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_329: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_328, 0.2);  mul_328 = None
        add_90: "f16[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_329, add_81);  mul_329 = add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_164: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(add_90, torch.float32);  add_90 = None
        mul_330: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.5)
        mul_331: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.7071067811865476);  convert_element_type_164 = None
        erf_39: "f32[128, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_331);  mul_331 = None
        add_91: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_332: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_330, add_91);  mul_330 = add_91 = None
        convert_element_type_165: "f16[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_332, torch.float16);  mul_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_333: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_165, 1.7015043497085571);  convert_element_type_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_334: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_333, 0.8980265101338745);  mul_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_44: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_132, getitem_89);  view_132 = getitem_89 = None
        add_93: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_339: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_338: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg179_1, 0.02551551815399144);  arg179_1 = None
        view_133: "f16[768]" = torch.ops.aten.reshape.default(mul_338, [-1]);  mul_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_44: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_133, -1);  view_133 = None
        mul_340: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_339, unsqueeze_44);  mul_339 = unsqueeze_44 = None
        convert_element_type_169: "f16[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(mul_340, torch.float16);  mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_134: "f16[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_169, [768, 1536, 1, 1]);  convert_element_type_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_62: "f16[128, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_334, view_134, arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_134 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_170: "f32[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32);  convolution_62 = None
        mul_341: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_170, 0.5)
        mul_342: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_170, 0.7071067811865476);  convert_element_type_170 = None
        erf_40: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_342);  mul_342 = None
        add_94: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_343: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_341, add_94);  mul_341 = add_94 = None
        convert_element_type_171: "f16[128, 768, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_343, torch.float16);  mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_344: "f16[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_171, 1.7015043497085571);  convert_element_type_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_4: "f16[128, 768, 13, 13]" = torch.ops.aten.constant_pad_nd.default(mul_344, [0, 1, 0, 1], 0.0);  mul_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_45: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_135, getitem_91);  view_135 = getitem_91 = None
        add_95: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_346: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_345: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg182_1, 0.02946278254943948);  arg182_1 = None
        view_136: "f16[768]" = torch.ops.aten.reshape.default(mul_345, [-1]);  mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_45: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_136, -1);  view_136 = None
        mul_347: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_45);  mul_346 = unsqueeze_45 = None
        convert_element_type_173: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_347, torch.float16);  mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_137: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_173, [768, 128, 3, 3]);  convert_element_type_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_63: "f16[128, 768, 6, 6]" = torch.ops.aten.convolution.default(constant_pad_nd_4, view_137, arg183_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6);  constant_pad_nd_4 = view_137 = arg183_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_174: "f32[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32);  convolution_63 = None
        mul_348: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_174, 0.5)
        mul_349: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_174, 0.7071067811865476);  convert_element_type_174 = None
        erf_41: "f32[128, 768, 6, 6]" = torch.ops.aten.erf.default(mul_349);  mul_349 = None
        add_96: "f32[128, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_350: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_348, add_96);  mul_348 = add_96 = None
        convert_element_type_175: "f16[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_350, torch.float16);  mul_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_351: "f16[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_175, 1.7015043497085571);  convert_element_type_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_46: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_138, getitem_93);  view_138 = getitem_93 = None
        add_97: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_353: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_352: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg185_1, 0.02946278254943948);  arg185_1 = None
        view_139: "f16[768]" = torch.ops.aten.reshape.default(mul_352, [-1]);  mul_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_46: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_139, -1);  view_139 = None
        mul_354: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_353, unsqueeze_46);  mul_353 = unsqueeze_46 = None
        convert_element_type_177: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_354, torch.float16);  mul_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_140: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_177, [768, 128, 3, 3]);  convert_element_type_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_64: "f16[128, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_351, view_140, arg186_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_351 = view_140 = arg186_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_178: "f32[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_64, torch.float32);  convolution_64 = None
        mul_355: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_178, 0.5)
        mul_356: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_178, 0.7071067811865476);  convert_element_type_178 = None
        erf_42: "f32[128, 768, 6, 6]" = torch.ops.aten.erf.default(mul_356);  mul_356 = None
        add_98: "f32[128, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_357: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_355, add_98);  mul_355 = add_98 = None
        convert_element_type_179: "f16[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_357, torch.float16);  mul_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_358: "f16[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_179, 1.7015043497085571);  convert_element_type_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_47: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_141, getitem_95);  view_141 = getitem_95 = None
        add_99: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_360: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_359: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg188_1, 0.03608439182435161);  arg188_1 = None
        view_142: "f16[1536]" = torch.ops.aten.reshape.default(mul_359, [-1]);  mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_47: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_142, -1);  view_142 = None
        mul_361: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_360, unsqueeze_47);  mul_360 = unsqueeze_47 = None
        convert_element_type_181: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_361, torch.float16);  mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_143: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_181, [1536, 768, 1, 1]);  convert_element_type_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_65: "f16[128, 1536, 6, 6]" = torch.ops.aten.convolution.default(mul_358, view_143, arg189_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_358 = view_143 = arg189_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_9: "f16[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_65, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_66: "f16[128, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_9, arg190_1, arg191_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg190_1 = arg191_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_9: "f16[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_67: "f16[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_9, arg192_1, arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg192_1 = arg193_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_9: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_362: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_9);  convolution_65 = sigmoid_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_363: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_362, 2.0);  mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_364: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_363, arg194_1);  mul_363 = arg194_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_365: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_364, 0.2);  mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_2: "f16[128, 1536, 6, 6]" = torch.ops.aten.avg_pool2d.default(mul_334, [2, 2], [2, 2], [0, 0], True, False);  mul_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_43: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, getitem_87);  view_129 = getitem_87 = None
        add_92: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_336: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_335: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg176_1, 0.02551551815399144);  arg176_1 = None
        view_130: "f16[1536]" = torch.ops.aten.reshape.default(mul_335, [-1]);  mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_43: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_130, -1);  view_130 = None
        mul_337: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_43);  mul_336 = unsqueeze_43 = None
        convert_element_type_167: "f16[1, 1536, 1536]" = torch.ops.prims.convert_element_type.default(mul_337, torch.float16);  mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_131: "f16[1536, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_167, [1536, 1536, 1, 1]);  convert_element_type_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_61: "f16[128, 1536, 6, 6]" = torch.ops.aten.convolution.default(avg_pool2d_2, view_131, arg177_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_2 = view_131 = arg177_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        add_100: "f16[128, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_365, convolution_61);  mul_365 = convolution_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_182: "f32[128, 1536, 6, 6]" = torch.ops.prims.convert_element_type.default(add_100, torch.float32)
        mul_366: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.5)
        mul_367: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.7071067811865476);  convert_element_type_182 = None
        erf_43: "f32[128, 1536, 6, 6]" = torch.ops.aten.erf.default(mul_367);  mul_367 = None
        add_101: "f32[128, 1536, 6, 6]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_368: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_366, add_101);  mul_366 = add_101 = None
        convert_element_type_183: "f16[128, 1536, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_368, torch.float16);  mul_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_369: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_183, 1.7015043497085571);  convert_element_type_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_370: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_369, 0.9805806756909201);  mul_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_48: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_144, getitem_97);  view_144 = getitem_97 = None
        add_102: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_372: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_371: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg196_1, 0.02551551815399144);  arg196_1 = None
        view_145: "f16[768]" = torch.ops.aten.reshape.default(mul_371, [-1]);  mul_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_48: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_145, -1);  view_145 = None
        mul_373: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_372, unsqueeze_48);  mul_372 = unsqueeze_48 = None
        convert_element_type_185: "f16[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(mul_373, torch.float16);  mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_146: "f16[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_185, [768, 1536, 1, 1]);  convert_element_type_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_68: "f16[128, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_370, view_146, arg197_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_370 = view_146 = arg197_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_186: "f32[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_68, torch.float32);  convolution_68 = None
        mul_374: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_186, 0.5)
        mul_375: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_186, 0.7071067811865476);  convert_element_type_186 = None
        erf_44: "f32[128, 768, 6, 6]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
        add_103: "f32[128, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_376: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_374, add_103);  mul_374 = add_103 = None
        convert_element_type_187: "f16[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_376, torch.float16);  mul_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_377: "f16[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_187, 1.7015043497085571);  convert_element_type_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_49: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_147, getitem_99);  view_147 = getitem_99 = None
        add_104: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_49: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_379: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_378: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg199_1, 0.02946278254943948);  arg199_1 = None
        view_148: "f16[768]" = torch.ops.aten.reshape.default(mul_378, [-1]);  mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_49: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_148, -1);  view_148 = None
        mul_380: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_49);  mul_379 = unsqueeze_49 = None
        convert_element_type_189: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_380, torch.float16);  mul_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_149: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_189, [768, 128, 3, 3]);  convert_element_type_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_69: "f16[128, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_377, view_149, arg200_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_377 = view_149 = arg200_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_190: "f32[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32);  convolution_69 = None
        mul_381: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_190, 0.5)
        mul_382: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_190, 0.7071067811865476);  convert_element_type_190 = None
        erf_45: "f32[128, 768, 6, 6]" = torch.ops.aten.erf.default(mul_382);  mul_382 = None
        add_105: "f32[128, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_383: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_381, add_105);  mul_381 = add_105 = None
        convert_element_type_191: "f16[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_383, torch.float16);  mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_384: "f16[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_191, 1.7015043497085571);  convert_element_type_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_50: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_150, getitem_101);  view_150 = getitem_101 = None
        add_106: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_50: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_386: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_385: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg202_1, 0.02946278254943948);  arg202_1 = None
        view_151: "f16[768]" = torch.ops.aten.reshape.default(mul_385, [-1]);  mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_50: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_151, -1);  view_151 = None
        mul_387: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_386, unsqueeze_50);  mul_386 = unsqueeze_50 = None
        convert_element_type_193: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_387, torch.float16);  mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_152: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_193, [768, 128, 3, 3]);  convert_element_type_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_70: "f16[128, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_384, view_152, arg203_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_384 = view_152 = arg203_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_194: "f32[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_70, torch.float32);  convolution_70 = None
        mul_388: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_194, 0.5)
        mul_389: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_194, 0.7071067811865476);  convert_element_type_194 = None
        erf_46: "f32[128, 768, 6, 6]" = torch.ops.aten.erf.default(mul_389);  mul_389 = None
        add_107: "f32[128, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_390: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_388, add_107);  mul_388 = add_107 = None
        convert_element_type_195: "f16[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_390, torch.float16);  mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_391: "f16[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_195, 1.7015043497085571);  convert_element_type_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_51: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_153, getitem_103);  view_153 = getitem_103 = None
        add_108: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_51: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_393: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_392: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg205_1, 0.03608439182435161);  arg205_1 = None
        view_154: "f16[1536]" = torch.ops.aten.reshape.default(mul_392, [-1]);  mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_51: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_154, -1);  view_154 = None
        mul_394: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_393, unsqueeze_51);  mul_393 = unsqueeze_51 = None
        convert_element_type_197: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_394, torch.float16);  mul_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_155: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_197, [1536, 768, 1, 1]);  convert_element_type_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_71: "f16[128, 1536, 6, 6]" = torch.ops.aten.convolution.default(mul_391, view_155, arg206_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_391 = view_155 = arg206_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_10: "f16[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_71, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_72: "f16[128, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_10, arg207_1, arg208_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg207_1 = arg208_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_10: "f16[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_72);  convolution_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_73: "f16[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_10, arg209_1, arg210_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_10 = arg209_1 = arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_10: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_395: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_10);  convolution_71 = sigmoid_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_396: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_395, 2.0);  mul_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_397: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_396, arg211_1);  mul_396 = arg211_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_398: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_397, 0.2);  mul_397 = None
        add_109: "f16[128, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_398, add_100);  mul_398 = add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_198: "f32[128, 1536, 6, 6]" = torch.ops.prims.convert_element_type.default(add_109, torch.float32)
        mul_399: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.5)
        mul_400: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.7071067811865476);  convert_element_type_198 = None
        erf_47: "f32[128, 1536, 6, 6]" = torch.ops.aten.erf.default(mul_400);  mul_400 = None
        add_110: "f32[128, 1536, 6, 6]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_401: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_399, add_110);  mul_399 = add_110 = None
        convert_element_type_199: "f16[128, 1536, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_401, torch.float16);  mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_402: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_199, 1.7015043497085571);  convert_element_type_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_403: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_402, 0.9622504486493761);  mul_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_52: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_156, getitem_105);  view_156 = getitem_105 = None
        add_111: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_52: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_405: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_404: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg213_1, 0.02551551815399144);  arg213_1 = None
        view_157: "f16[768]" = torch.ops.aten.reshape.default(mul_404, [-1]);  mul_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_52: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_157, -1);  view_157 = None
        mul_406: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_52);  mul_405 = unsqueeze_52 = None
        convert_element_type_201: "f16[1, 768, 1536]" = torch.ops.prims.convert_element_type.default(mul_406, torch.float16);  mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_158: "f16[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_201, [768, 1536, 1, 1]);  convert_element_type_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_74: "f16[128, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_403, view_158, arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_403 = view_158 = arg214_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_202: "f32[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_74, torch.float32);  convolution_74 = None
        mul_407: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_202, 0.5)
        mul_408: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_202, 0.7071067811865476);  convert_element_type_202 = None
        erf_48: "f32[128, 768, 6, 6]" = torch.ops.aten.erf.default(mul_408);  mul_408 = None
        add_112: "f32[128, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_409: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_407, add_112);  mul_407 = add_112 = None
        convert_element_type_203: "f16[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_409, torch.float16);  mul_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_410: "f16[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_203, 1.7015043497085571);  convert_element_type_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_53: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_159, getitem_107);  view_159 = getitem_107 = None
        add_113: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
        rsqrt_53: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_412: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_411: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg216_1, 0.02946278254943948);  arg216_1 = None
        view_160: "f16[768]" = torch.ops.aten.reshape.default(mul_411, [-1]);  mul_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_53: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_160, -1);  view_160 = None
        mul_413: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_53);  mul_412 = unsqueeze_53 = None
        convert_element_type_205: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_413, torch.float16);  mul_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_161: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_205, [768, 128, 3, 3]);  convert_element_type_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_75: "f16[128, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_410, view_161, arg217_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_410 = view_161 = arg217_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_206: "f32[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32);  convolution_75 = None
        mul_414: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_206, 0.5)
        mul_415: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_206, 0.7071067811865476);  convert_element_type_206 = None
        erf_49: "f32[128, 768, 6, 6]" = torch.ops.aten.erf.default(mul_415);  mul_415 = None
        add_114: "f32[128, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_416: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_414, add_114);  mul_414 = add_114 = None
        convert_element_type_207: "f16[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_416, torch.float16);  mul_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_417: "f16[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_207, 1.7015043497085571);  convert_element_type_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_54: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_162, getitem_109);  view_162 = getitem_109 = None
        add_115: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_54: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_419: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_418: "f16[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg219_1, 0.02946278254943948);  arg219_1 = None
        view_163: "f16[768]" = torch.ops.aten.reshape.default(mul_418, [-1]);  mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_54: "f16[768, 1]" = torch.ops.aten.unsqueeze.default(view_163, -1);  view_163 = None
        mul_420: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_54);  mul_419 = unsqueeze_54 = None
        convert_element_type_209: "f16[1, 768, 1152]" = torch.ops.prims.convert_element_type.default(mul_420, torch.float16);  mul_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_164: "f16[768, 128, 3, 3]" = torch.ops.aten.reshape.default(convert_element_type_209, [768, 128, 3, 3]);  convert_element_type_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_76: "f16[128, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_417, view_164, arg220_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_417 = view_164 = arg220_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_210: "f32[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32);  convolution_76 = None
        mul_421: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.5)
        mul_422: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.7071067811865476);  convert_element_type_210 = None
        erf_50: "f32[128, 768, 6, 6]" = torch.ops.aten.erf.default(mul_422);  mul_422 = None
        add_116: "f32[128, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_423: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_421, add_116);  mul_421 = add_116 = None
        convert_element_type_211: "f16[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_423, torch.float16);  mul_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_424: "f16[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_211, 1.7015043497085571);  convert_element_type_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_55: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_165, getitem_111);  view_165 = getitem_111 = None
        add_117: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_55: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        mul_426: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_425: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg222_1, 0.03608439182435161);  arg222_1 = None
        view_166: "f16[1536]" = torch.ops.aten.reshape.default(mul_425, [-1]);  mul_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_55: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_166, -1);  view_166 = None
        mul_427: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_55);  mul_426 = unsqueeze_55 = None
        convert_element_type_213: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_427, torch.float16);  mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_167: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_213, [1536, 768, 1, 1]);  convert_element_type_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_77: "f16[128, 1536, 6, 6]" = torch.ops.aten.convolution.default(mul_424, view_167, arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_424 = view_167 = arg223_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_11: "f16[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_77, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_78: "f16[128, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_11, arg224_1, arg225_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg224_1 = arg225_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_11: "f16[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_78);  convolution_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_79: "f16[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_11, arg226_1, arg227_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg226_1 = arg227_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_11: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_428: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_11);  convolution_77 = sigmoid_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_429: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_428, 2.0);  mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_430: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_429, arg228_1);  mul_429 = arg228_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_431: "f16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_430, 0.2);  mul_430 = None
        add_118: "f16[128, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_431, add_109);  mul_431 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_56: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view_168, getitem_113);  view_168 = getitem_113 = None
        add_119: "f32[1, 3072, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_56: "f32[1, 3072, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_433: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_432: "f16[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg230_1, 0.02551551815399144);  arg230_1 = None
        view_169: "f16[3072]" = torch.ops.aten.reshape.default(mul_432, [-1]);  mul_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_56: "f16[3072, 1]" = torch.ops.aten.unsqueeze.default(view_169, -1);  view_169 = None
        mul_434: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_56);  mul_433 = unsqueeze_56 = None
        convert_element_type_215: "f16[1, 3072, 1536]" = torch.ops.prims.convert_element_type.default(mul_434, torch.float16);  mul_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        view_170: "f16[3072, 1536, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_215, [3072, 1536, 1, 1]);  convert_element_type_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_80: "f16[128, 3072, 6, 6]" = torch.ops.aten.convolution.default(add_118, view_170, arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_118 = view_170 = arg231_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_216: "f32[128, 3072, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32);  convolution_80 = None
        mul_435: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_216, 0.5)
        mul_436: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_216, 0.7071067811865476);  convert_element_type_216 = None
        erf_51: "f32[128, 3072, 6, 6]" = torch.ops.aten.erf.default(mul_436);  mul_436 = None
        add_120: "f32[128, 3072, 6, 6]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_437: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_435, add_120);  mul_435 = add_120 = None
        convert_element_type_217: "f16[128, 3072, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_437, torch.float16);  mul_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_438: "f16[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_217, 1.7015043497085571);  convert_element_type_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_12: "f16[128, 3072, 1, 1]" = torch.ops.aten.mean.dim(mul_438, [-1, -2], True);  mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_171: "f16[128, 3072]" = torch.ops.aten.reshape.default(mean_12, [128, 3072]);  mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute: "f16[3072, 1000]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm: "f16[128, 1000]" = torch.ops.aten.addmm.default(arg233_1, view_171, permute);  arg233_1 = view_171 = permute = None
        return (addmm,)

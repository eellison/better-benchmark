class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[16, 1, 1, 1]", arg2_1: "f32[16]", arg3_1: "f32[128, 3, 224, 224]", arg4_1: "f32[32, 16, 3, 3]", arg5_1: "f32[32, 1, 1, 1]", arg6_1: "f32[32]", arg7_1: "f32[64, 32, 3, 3]", arg8_1: "f32[64, 1, 1, 1]", arg9_1: "f32[64]", arg10_1: "f32[128, 64, 3, 3]", arg11_1: "f32[128, 1, 1, 1]", arg12_1: "f32[128]", arg13_1: "f32[256, 128, 1, 1]", arg14_1: "f32[256, 1, 1, 1]", arg15_1: "f32[256]", arg16_1: "f32[64, 128, 1, 1]", arg17_1: "f32[64, 1, 1, 1]", arg18_1: "f32[64]", arg19_1: "f32[64, 64, 3, 3]", arg20_1: "f32[64, 1, 1, 1]", arg21_1: "f32[64]", arg22_1: "f32[64, 64, 3, 3]", arg23_1: "f32[64, 1, 1, 1]", arg24_1: "f32[64]", arg25_1: "f32[256, 64, 1, 1]", arg26_1: "f32[256, 1, 1, 1]", arg27_1: "f32[256]", arg28_1: "f32[64, 256, 1, 1]", arg29_1: "f32[64]", arg30_1: "f32[256, 64, 1, 1]", arg31_1: "f32[256]", arg32_1: "f32[512, 256, 1, 1]", arg33_1: "f32[512, 1, 1, 1]", arg34_1: "f32[512]", arg35_1: "f32[128, 256, 1, 1]", arg36_1: "f32[128, 1, 1, 1]", arg37_1: "f32[128]", arg38_1: "f32[128, 64, 3, 3]", arg39_1: "f32[128, 1, 1, 1]", arg40_1: "f32[128]", arg41_1: "f32[128, 64, 3, 3]", arg42_1: "f32[128, 1, 1, 1]", arg43_1: "f32[128]", arg44_1: "f32[512, 128, 1, 1]", arg45_1: "f32[512, 1, 1, 1]", arg46_1: "f32[512]", arg47_1: "f32[128, 512, 1, 1]", arg48_1: "f32[128]", arg49_1: "f32[512, 128, 1, 1]", arg50_1: "f32[512]", arg51_1: "f32[128, 512, 1, 1]", arg52_1: "f32[128, 1, 1, 1]", arg53_1: "f32[128]", arg54_1: "f32[128, 64, 3, 3]", arg55_1: "f32[128, 1, 1, 1]", arg56_1: "f32[128]", arg57_1: "f32[128, 64, 3, 3]", arg58_1: "f32[128, 1, 1, 1]", arg59_1: "f32[128]", arg60_1: "f32[512, 128, 1, 1]", arg61_1: "f32[512, 1, 1, 1]", arg62_1: "f32[512]", arg63_1: "f32[128, 512, 1, 1]", arg64_1: "f32[128]", arg65_1: "f32[512, 128, 1, 1]", arg66_1: "f32[512]", arg67_1: "f32[1536, 512, 1, 1]", arg68_1: "f32[1536, 1, 1, 1]", arg69_1: "f32[1536]", arg70_1: "f32[384, 512, 1, 1]", arg71_1: "f32[384, 1, 1, 1]", arg72_1: "f32[384]", arg73_1: "f32[384, 64, 3, 3]", arg74_1: "f32[384, 1, 1, 1]", arg75_1: "f32[384]", arg76_1: "f32[384, 64, 3, 3]", arg77_1: "f32[384, 1, 1, 1]", arg78_1: "f32[384]", arg79_1: "f32[1536, 384, 1, 1]", arg80_1: "f32[1536, 1, 1, 1]", arg81_1: "f32[1536]", arg82_1: "f32[384, 1536, 1, 1]", arg83_1: "f32[384]", arg84_1: "f32[1536, 384, 1, 1]", arg85_1: "f32[1536]", arg86_1: "f32[384, 1536, 1, 1]", arg87_1: "f32[384, 1, 1, 1]", arg88_1: "f32[384]", arg89_1: "f32[384, 64, 3, 3]", arg90_1: "f32[384, 1, 1, 1]", arg91_1: "f32[384]", arg92_1: "f32[384, 64, 3, 3]", arg93_1: "f32[384, 1, 1, 1]", arg94_1: "f32[384]", arg95_1: "f32[1536, 384, 1, 1]", arg96_1: "f32[1536, 1, 1, 1]", arg97_1: "f32[1536]", arg98_1: "f32[384, 1536, 1, 1]", arg99_1: "f32[384]", arg100_1: "f32[1536, 384, 1, 1]", arg101_1: "f32[1536]", arg102_1: "f32[384, 1536, 1, 1]", arg103_1: "f32[384, 1, 1, 1]", arg104_1: "f32[384]", arg105_1: "f32[384, 64, 3, 3]", arg106_1: "f32[384, 1, 1, 1]", arg107_1: "f32[384]", arg108_1: "f32[384, 64, 3, 3]", arg109_1: "f32[384, 1, 1, 1]", arg110_1: "f32[384]", arg111_1: "f32[1536, 384, 1, 1]", arg112_1: "f32[1536, 1, 1, 1]", arg113_1: "f32[1536]", arg114_1: "f32[384, 1536, 1, 1]", arg115_1: "f32[384]", arg116_1: "f32[1536, 384, 1, 1]", arg117_1: "f32[1536]", arg118_1: "f32[384, 1536, 1, 1]", arg119_1: "f32[384, 1, 1, 1]", arg120_1: "f32[384]", arg121_1: "f32[384, 64, 3, 3]", arg122_1: "f32[384, 1, 1, 1]", arg123_1: "f32[384]", arg124_1: "f32[384, 64, 3, 3]", arg125_1: "f32[384, 1, 1, 1]", arg126_1: "f32[384]", arg127_1: "f32[1536, 384, 1, 1]", arg128_1: "f32[1536, 1, 1, 1]", arg129_1: "f32[1536]", arg130_1: "f32[384, 1536, 1, 1]", arg131_1: "f32[384]", arg132_1: "f32[1536, 384, 1, 1]", arg133_1: "f32[1536]", arg134_1: "f32[384, 1536, 1, 1]", arg135_1: "f32[384, 1, 1, 1]", arg136_1: "f32[384]", arg137_1: "f32[384, 64, 3, 3]", arg138_1: "f32[384, 1, 1, 1]", arg139_1: "f32[384]", arg140_1: "f32[384, 64, 3, 3]", arg141_1: "f32[384, 1, 1, 1]", arg142_1: "f32[384]", arg143_1: "f32[1536, 384, 1, 1]", arg144_1: "f32[1536, 1, 1, 1]", arg145_1: "f32[1536]", arg146_1: "f32[384, 1536, 1, 1]", arg147_1: "f32[384]", arg148_1: "f32[1536, 384, 1, 1]", arg149_1: "f32[1536]", arg150_1: "f32[384, 1536, 1, 1]", arg151_1: "f32[384, 1, 1, 1]", arg152_1: "f32[384]", arg153_1: "f32[384, 64, 3, 3]", arg154_1: "f32[384, 1, 1, 1]", arg155_1: "f32[384]", arg156_1: "f32[384, 64, 3, 3]", arg157_1: "f32[384, 1, 1, 1]", arg158_1: "f32[384]", arg159_1: "f32[1536, 384, 1, 1]", arg160_1: "f32[1536, 1, 1, 1]", arg161_1: "f32[1536]", arg162_1: "f32[384, 1536, 1, 1]", arg163_1: "f32[384]", arg164_1: "f32[1536, 384, 1, 1]", arg165_1: "f32[1536]", arg166_1: "f32[1536, 1536, 1, 1]", arg167_1: "f32[1536, 1, 1, 1]", arg168_1: "f32[1536]", arg169_1: "f32[384, 1536, 1, 1]", arg170_1: "f32[384, 1, 1, 1]", arg171_1: "f32[384]", arg172_1: "f32[384, 64, 3, 3]", arg173_1: "f32[384, 1, 1, 1]", arg174_1: "f32[384]", arg175_1: "f32[384, 64, 3, 3]", arg176_1: "f32[384, 1, 1, 1]", arg177_1: "f32[384]", arg178_1: "f32[1536, 384, 1, 1]", arg179_1: "f32[1536, 1, 1, 1]", arg180_1: "f32[1536]", arg181_1: "f32[384, 1536, 1, 1]", arg182_1: "f32[384]", arg183_1: "f32[1536, 384, 1, 1]", arg184_1: "f32[1536]", arg185_1: "f32[384, 1536, 1, 1]", arg186_1: "f32[384, 1, 1, 1]", arg187_1: "f32[384]", arg188_1: "f32[384, 64, 3, 3]", arg189_1: "f32[384, 1, 1, 1]", arg190_1: "f32[384]", arg191_1: "f32[384, 64, 3, 3]", arg192_1: "f32[384, 1, 1, 1]", arg193_1: "f32[384]", arg194_1: "f32[1536, 384, 1, 1]", arg195_1: "f32[1536, 1, 1, 1]", arg196_1: "f32[1536]", arg197_1: "f32[384, 1536, 1, 1]", arg198_1: "f32[384]", arg199_1: "f32[1536, 384, 1, 1]", arg200_1: "f32[1536]", arg201_1: "f32[384, 1536, 1, 1]", arg202_1: "f32[384, 1, 1, 1]", arg203_1: "f32[384]", arg204_1: "f32[384, 64, 3, 3]", arg205_1: "f32[384, 1, 1, 1]", arg206_1: "f32[384]", arg207_1: "f32[384, 64, 3, 3]", arg208_1: "f32[384, 1, 1, 1]", arg209_1: "f32[384]", arg210_1: "f32[1536, 384, 1, 1]", arg211_1: "f32[1536, 1, 1, 1]", arg212_1: "f32[1536]", arg213_1: "f32[384, 1536, 1, 1]", arg214_1: "f32[384]", arg215_1: "f32[1536, 384, 1, 1]", arg216_1: "f32[1536]", arg217_1: "f32[2304, 1536, 1, 1]", arg218_1: "f32[2304, 1, 1, 1]", arg219_1: "f32[2304]", arg220_1: "f32[1000, 2304]", arg221_1: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone: "f32[16, 3, 3, 3]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(clone, [1, 16, 27]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean = torch.ops.aten.var_mean.correction(view, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1]" = var_mean[0]
        getitem_1: "f32[1, 16, 1]" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_1: "f32[32, 16, 3, 3]" = torch.ops.aten.clone.default(arg4_1, memory_format = torch.contiguous_format);  arg4_1 = None
        view_3: "f32[1, 32, 144]" = torch.ops.aten.reshape.default(clone_1, [1, 32, 144]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [0, 2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 32, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 32, 1]" = var_mean_1[1];  var_mean_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_2: "f32[64, 32, 3, 3]" = torch.ops.aten.clone.default(arg7_1, memory_format = torch.contiguous_format);  arg7_1 = None
        view_6: "f32[1, 64, 288]" = torch.ops.aten.reshape.default(clone_2, [1, 64, 288]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [0, 2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1]" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_3: "f32[128, 64, 3, 3]" = torch.ops.aten.clone.default(arg10_1, memory_format = torch.contiguous_format);  arg10_1 = None
        view_9: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(clone_3, [1, 128, 576]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_3 = torch.ops.aten.var_mean.correction(view_9, [0, 2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_12: "f32[1, 256, 128]" = torch.ops.aten.reshape.default(arg13_1, [1, 256, -1]);  arg13_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_4 = torch.ops.aten.var_mean.correction(view_12, [0, 2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 256, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 256, 1]" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_15: "f32[1, 64, 128]" = torch.ops.aten.reshape.default(arg16_1, [1, 64, -1]);  arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [0, 2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 64, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 64, 1]" = var_mean_5[1];  var_mean_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_4: "f32[64, 64, 3, 3]" = torch.ops.aten.clone.default(arg19_1, memory_format = torch.contiguous_format);  arg19_1 = None
        view_18: "f32[1, 64, 576]" = torch.ops.aten.reshape.default(clone_4, [1, 64, 576]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_6 = torch.ops.aten.var_mean.correction(view_18, [0, 2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 64, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 64, 1]" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_5: "f32[64, 64, 3, 3]" = torch.ops.aten.clone.default(arg22_1, memory_format = torch.contiguous_format);  arg22_1 = None
        view_21: "f32[1, 64, 576]" = torch.ops.aten.reshape.default(clone_5, [1, 64, 576]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_7 = torch.ops.aten.var_mean.correction(view_21, [0, 2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 64, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 64, 1]" = var_mean_7[1];  var_mean_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_24: "f32[1, 256, 64]" = torch.ops.aten.reshape.default(arg25_1, [1, 256, -1]);  arg25_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_8 = torch.ops.aten.var_mean.correction(view_24, [0, 2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 256, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 256, 1]" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_27: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(arg32_1, [1, 512, -1]);  arg32_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_9 = torch.ops.aten.var_mean.correction(view_27, [0, 2], correction = 0, keepdim = True)
        getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_30: "f32[1, 128, 256]" = torch.ops.aten.reshape.default(arg35_1, [1, 128, -1]);  arg35_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_10 = torch.ops.aten.var_mean.correction(view_30, [0, 2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_6: "f32[128, 64, 3, 3]" = torch.ops.aten.clone.default(arg38_1, memory_format = torch.contiguous_format);  arg38_1 = None
        view_33: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(clone_6, [1, 128, 576]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_11 = torch.ops.aten.var_mean.correction(view_33, [0, 2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_7: "f32[128, 64, 3, 3]" = torch.ops.aten.clone.default(arg41_1, memory_format = torch.contiguous_format);  arg41_1 = None
        view_36: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(clone_7, [1, 128, 576]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_12 = torch.ops.aten.var_mean.correction(view_36, [0, 2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_39: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(arg44_1, [1, 512, -1]);  arg44_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_13 = torch.ops.aten.var_mean.correction(view_39, [0, 2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_42: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(arg51_1, [1, 128, -1]);  arg51_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_14 = torch.ops.aten.var_mean.correction(view_42, [0, 2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_8: "f32[128, 64, 3, 3]" = torch.ops.aten.clone.default(arg54_1, memory_format = torch.contiguous_format);  arg54_1 = None
        view_45: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(clone_8, [1, 128, 576]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_15 = torch.ops.aten.var_mean.correction(view_45, [0, 2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_9: "f32[128, 64, 3, 3]" = torch.ops.aten.clone.default(arg57_1, memory_format = torch.contiguous_format);  arg57_1 = None
        view_48: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(clone_9, [1, 128, 576]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_16 = torch.ops.aten.var_mean.correction(view_48, [0, 2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_51: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(arg60_1, [1, 512, -1]);  arg60_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_17 = torch.ops.aten.var_mean.correction(view_51, [0, 2], correction = 0, keepdim = True)
        getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_54: "f32[1, 1536, 512]" = torch.ops.aten.reshape.default(arg67_1, [1, 1536, -1]);  arg67_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_18 = torch.ops.aten.var_mean.correction(view_54, [0, 2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 1536, 1]" = var_mean_18[0]
        getitem_37: "f32[1, 1536, 1]" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_57: "f32[1, 384, 512]" = torch.ops.aten.reshape.default(arg70_1, [1, 384, -1]);  arg70_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_19 = torch.ops.aten.var_mean.correction(view_57, [0, 2], correction = 0, keepdim = True)
        getitem_38: "f32[1, 384, 1]" = var_mean_19[0]
        getitem_39: "f32[1, 384, 1]" = var_mean_19[1];  var_mean_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_10: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg73_1, memory_format = torch.contiguous_format);  arg73_1 = None
        view_60: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_10, [1, 384, 576]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_20 = torch.ops.aten.var_mean.correction(view_60, [0, 2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 384, 1]" = var_mean_20[0]
        getitem_41: "f32[1, 384, 1]" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_11: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg76_1, memory_format = torch.contiguous_format);  arg76_1 = None
        view_63: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_11, [1, 384, 576]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_21 = torch.ops.aten.var_mean.correction(view_63, [0, 2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 384, 1]" = var_mean_21[0]
        getitem_43: "f32[1, 384, 1]" = var_mean_21[1];  var_mean_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_66: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg79_1, [1, 1536, -1]);  arg79_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_22 = torch.ops.aten.var_mean.correction(view_66, [0, 2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 1536, 1]" = var_mean_22[0]
        getitem_45: "f32[1, 1536, 1]" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_69: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(arg86_1, [1, 384, -1]);  arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_23 = torch.ops.aten.var_mean.correction(view_69, [0, 2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 384, 1]" = var_mean_23[0]
        getitem_47: "f32[1, 384, 1]" = var_mean_23[1];  var_mean_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_12: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg89_1, memory_format = torch.contiguous_format);  arg89_1 = None
        view_72: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_12, [1, 384, 576]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_24 = torch.ops.aten.var_mean.correction(view_72, [0, 2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 384, 1]" = var_mean_24[0]
        getitem_49: "f32[1, 384, 1]" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_13: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg92_1, memory_format = torch.contiguous_format);  arg92_1 = None
        view_75: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_13, [1, 384, 576]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_25 = torch.ops.aten.var_mean.correction(view_75, [0, 2], correction = 0, keepdim = True)
        getitem_50: "f32[1, 384, 1]" = var_mean_25[0]
        getitem_51: "f32[1, 384, 1]" = var_mean_25[1];  var_mean_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_78: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg95_1, [1, 1536, -1]);  arg95_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_26 = torch.ops.aten.var_mean.correction(view_78, [0, 2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 1536, 1]" = var_mean_26[0]
        getitem_53: "f32[1, 1536, 1]" = var_mean_26[1];  var_mean_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_81: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(arg102_1, [1, 384, -1]);  arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_27 = torch.ops.aten.var_mean.correction(view_81, [0, 2], correction = 0, keepdim = True)
        getitem_54: "f32[1, 384, 1]" = var_mean_27[0]
        getitem_55: "f32[1, 384, 1]" = var_mean_27[1];  var_mean_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_14: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg105_1, memory_format = torch.contiguous_format);  arg105_1 = None
        view_84: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_14, [1, 384, 576]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_28 = torch.ops.aten.var_mean.correction(view_84, [0, 2], correction = 0, keepdim = True)
        getitem_56: "f32[1, 384, 1]" = var_mean_28[0]
        getitem_57: "f32[1, 384, 1]" = var_mean_28[1];  var_mean_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_15: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg108_1, memory_format = torch.contiguous_format);  arg108_1 = None
        view_87: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_15, [1, 384, 576]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_29 = torch.ops.aten.var_mean.correction(view_87, [0, 2], correction = 0, keepdim = True)
        getitem_58: "f32[1, 384, 1]" = var_mean_29[0]
        getitem_59: "f32[1, 384, 1]" = var_mean_29[1];  var_mean_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_90: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg111_1, [1, 1536, -1]);  arg111_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_30 = torch.ops.aten.var_mean.correction(view_90, [0, 2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 1536, 1]" = var_mean_30[0]
        getitem_61: "f32[1, 1536, 1]" = var_mean_30[1];  var_mean_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_93: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(arg118_1, [1, 384, -1]);  arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_31 = torch.ops.aten.var_mean.correction(view_93, [0, 2], correction = 0, keepdim = True)
        getitem_62: "f32[1, 384, 1]" = var_mean_31[0]
        getitem_63: "f32[1, 384, 1]" = var_mean_31[1];  var_mean_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_16: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg121_1, memory_format = torch.contiguous_format);  arg121_1 = None
        view_96: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_16, [1, 384, 576]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_32 = torch.ops.aten.var_mean.correction(view_96, [0, 2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 384, 1]" = var_mean_32[0]
        getitem_65: "f32[1, 384, 1]" = var_mean_32[1];  var_mean_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_17: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg124_1, memory_format = torch.contiguous_format);  arg124_1 = None
        view_99: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_17, [1, 384, 576]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_33 = torch.ops.aten.var_mean.correction(view_99, [0, 2], correction = 0, keepdim = True)
        getitem_66: "f32[1, 384, 1]" = var_mean_33[0]
        getitem_67: "f32[1, 384, 1]" = var_mean_33[1];  var_mean_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_102: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg127_1, [1, 1536, -1]);  arg127_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_34 = torch.ops.aten.var_mean.correction(view_102, [0, 2], correction = 0, keepdim = True)
        getitem_68: "f32[1, 1536, 1]" = var_mean_34[0]
        getitem_69: "f32[1, 1536, 1]" = var_mean_34[1];  var_mean_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_105: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(arg134_1, [1, 384, -1]);  arg134_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_35 = torch.ops.aten.var_mean.correction(view_105, [0, 2], correction = 0, keepdim = True)
        getitem_70: "f32[1, 384, 1]" = var_mean_35[0]
        getitem_71: "f32[1, 384, 1]" = var_mean_35[1];  var_mean_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_18: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg137_1, memory_format = torch.contiguous_format);  arg137_1 = None
        view_108: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_18, [1, 384, 576]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_36 = torch.ops.aten.var_mean.correction(view_108, [0, 2], correction = 0, keepdim = True)
        getitem_72: "f32[1, 384, 1]" = var_mean_36[0]
        getitem_73: "f32[1, 384, 1]" = var_mean_36[1];  var_mean_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_19: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg140_1, memory_format = torch.contiguous_format);  arg140_1 = None
        view_111: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_19, [1, 384, 576]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_37 = torch.ops.aten.var_mean.correction(view_111, [0, 2], correction = 0, keepdim = True)
        getitem_74: "f32[1, 384, 1]" = var_mean_37[0]
        getitem_75: "f32[1, 384, 1]" = var_mean_37[1];  var_mean_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_114: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg143_1, [1, 1536, -1]);  arg143_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_38 = torch.ops.aten.var_mean.correction(view_114, [0, 2], correction = 0, keepdim = True)
        getitem_76: "f32[1, 1536, 1]" = var_mean_38[0]
        getitem_77: "f32[1, 1536, 1]" = var_mean_38[1];  var_mean_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_117: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(arg150_1, [1, 384, -1]);  arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_39 = torch.ops.aten.var_mean.correction(view_117, [0, 2], correction = 0, keepdim = True)
        getitem_78: "f32[1, 384, 1]" = var_mean_39[0]
        getitem_79: "f32[1, 384, 1]" = var_mean_39[1];  var_mean_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_20: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg153_1, memory_format = torch.contiguous_format);  arg153_1 = None
        view_120: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_20, [1, 384, 576]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_40 = torch.ops.aten.var_mean.correction(view_120, [0, 2], correction = 0, keepdim = True)
        getitem_80: "f32[1, 384, 1]" = var_mean_40[0]
        getitem_81: "f32[1, 384, 1]" = var_mean_40[1];  var_mean_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_21: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg156_1, memory_format = torch.contiguous_format);  arg156_1 = None
        view_123: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_21, [1, 384, 576]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_41 = torch.ops.aten.var_mean.correction(view_123, [0, 2], correction = 0, keepdim = True)
        getitem_82: "f32[1, 384, 1]" = var_mean_41[0]
        getitem_83: "f32[1, 384, 1]" = var_mean_41[1];  var_mean_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_126: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg159_1, [1, 1536, -1]);  arg159_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_42 = torch.ops.aten.var_mean.correction(view_126, [0, 2], correction = 0, keepdim = True)
        getitem_84: "f32[1, 1536, 1]" = var_mean_42[0]
        getitem_85: "f32[1, 1536, 1]" = var_mean_42[1];  var_mean_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_129: "f32[1, 1536, 1536]" = torch.ops.aten.reshape.default(arg166_1, [1, 1536, -1]);  arg166_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_43 = torch.ops.aten.var_mean.correction(view_129, [0, 2], correction = 0, keepdim = True)
        getitem_86: "f32[1, 1536, 1]" = var_mean_43[0]
        getitem_87: "f32[1, 1536, 1]" = var_mean_43[1];  var_mean_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_132: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(arg169_1, [1, 384, -1]);  arg169_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_44 = torch.ops.aten.var_mean.correction(view_132, [0, 2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 384, 1]" = var_mean_44[0]
        getitem_89: "f32[1, 384, 1]" = var_mean_44[1];  var_mean_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_22: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg172_1, memory_format = torch.contiguous_format);  arg172_1 = None
        view_135: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_22, [1, 384, 576]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_45 = torch.ops.aten.var_mean.correction(view_135, [0, 2], correction = 0, keepdim = True)
        getitem_90: "f32[1, 384, 1]" = var_mean_45[0]
        getitem_91: "f32[1, 384, 1]" = var_mean_45[1];  var_mean_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_23: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg175_1, memory_format = torch.contiguous_format);  arg175_1 = None
        view_138: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_23, [1, 384, 576]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_46 = torch.ops.aten.var_mean.correction(view_138, [0, 2], correction = 0, keepdim = True)
        getitem_92: "f32[1, 384, 1]" = var_mean_46[0]
        getitem_93: "f32[1, 384, 1]" = var_mean_46[1];  var_mean_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_141: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg178_1, [1, 1536, -1]);  arg178_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_47 = torch.ops.aten.var_mean.correction(view_141, [0, 2], correction = 0, keepdim = True)
        getitem_94: "f32[1, 1536, 1]" = var_mean_47[0]
        getitem_95: "f32[1, 1536, 1]" = var_mean_47[1];  var_mean_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_144: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(arg185_1, [1, 384, -1]);  arg185_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_48 = torch.ops.aten.var_mean.correction(view_144, [0, 2], correction = 0, keepdim = True)
        getitem_96: "f32[1, 384, 1]" = var_mean_48[0]
        getitem_97: "f32[1, 384, 1]" = var_mean_48[1];  var_mean_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_24: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg188_1, memory_format = torch.contiguous_format);  arg188_1 = None
        view_147: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_24, [1, 384, 576]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_49 = torch.ops.aten.var_mean.correction(view_147, [0, 2], correction = 0, keepdim = True)
        getitem_98: "f32[1, 384, 1]" = var_mean_49[0]
        getitem_99: "f32[1, 384, 1]" = var_mean_49[1];  var_mean_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_25: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg191_1, memory_format = torch.contiguous_format);  arg191_1 = None
        view_150: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_25, [1, 384, 576]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_50 = torch.ops.aten.var_mean.correction(view_150, [0, 2], correction = 0, keepdim = True)
        getitem_100: "f32[1, 384, 1]" = var_mean_50[0]
        getitem_101: "f32[1, 384, 1]" = var_mean_50[1];  var_mean_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_153: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg194_1, [1, 1536, -1]);  arg194_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_51 = torch.ops.aten.var_mean.correction(view_153, [0, 2], correction = 0, keepdim = True)
        getitem_102: "f32[1, 1536, 1]" = var_mean_51[0]
        getitem_103: "f32[1, 1536, 1]" = var_mean_51[1];  var_mean_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_156: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(arg201_1, [1, 384, -1]);  arg201_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_52 = torch.ops.aten.var_mean.correction(view_156, [0, 2], correction = 0, keepdim = True)
        getitem_104: "f32[1, 384, 1]" = var_mean_52[0]
        getitem_105: "f32[1, 384, 1]" = var_mean_52[1];  var_mean_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_26: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg204_1, memory_format = torch.contiguous_format);  arg204_1 = None
        view_159: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_26, [1, 384, 576]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_53 = torch.ops.aten.var_mean.correction(view_159, [0, 2], correction = 0, keepdim = True)
        getitem_106: "f32[1, 384, 1]" = var_mean_53[0]
        getitem_107: "f32[1, 384, 1]" = var_mean_53[1];  var_mean_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_27: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg207_1, memory_format = torch.contiguous_format);  arg207_1 = None
        view_162: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_27, [1, 384, 576]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_54 = torch.ops.aten.var_mean.correction(view_162, [0, 2], correction = 0, keepdim = True)
        getitem_108: "f32[1, 384, 1]" = var_mean_54[0]
        getitem_109: "f32[1, 384, 1]" = var_mean_54[1];  var_mean_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_165: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg210_1, [1, 1536, -1]);  arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_55 = torch.ops.aten.var_mean.correction(view_165, [0, 2], correction = 0, keepdim = True)
        getitem_110: "f32[1, 1536, 1]" = var_mean_55[0]
        getitem_111: "f32[1, 1536, 1]" = var_mean_55[1];  var_mean_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_168: "f32[1, 2304, 1536]" = torch.ops.aten.reshape.default(arg217_1, [1, 2304, -1]);  arg217_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_56 = torch.ops.aten.var_mean.correction(view_168, [0, 2], correction = 0, keepdim = True)
        getitem_112: "f32[1, 2304, 1]" = var_mean_56[0]
        getitem_113: "f32[1, 2304, 1]" = var_mean_56[1];  var_mean_56 = None
        sub: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        add: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul_1: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg1_1, 0.34412564994580647);  arg1_1 = None
        view_1: "f32[16]" = torch.ops.aten.reshape.default(mul, [-1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
        mul_2: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze);  mul_1 = unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_2: "f32[16, 3, 3, 3]" = torch.ops.aten.reshape.default(mul_2, [16, 3, 3, 3]);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution: "f32[128, 16, 112, 112]" = torch.ops.aten.convolution.default(arg3_1, view_2, arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg3_1 = view_2 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        neg: "f32[128, 16, 112, 112]" = torch.ops.aten.neg.default(convolution)
        exp: "f32[128, 16, 112, 112]" = torch.ops.aten.exp.default(neg);  neg = None
        add_1: "f32[128, 16, 112, 112]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 16, 112, 112]" = torch.ops.aten.div.Tensor(convolution, add_1);  convolution = add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_1: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, getitem_3);  view_3 = getitem_3 = None
        add_2: "f32[1, 32, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[1, 32, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_3: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg5_1, 0.1490107774734497);  arg5_1 = None
        view_4: "f32[32]" = torch.ops.aten.reshape.default(mul_3, [-1]);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_1: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(view_4, -1);  view_4 = None
        mul_5: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_1);  mul_4 = unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_5: "f32[32, 16, 3, 3]" = torch.ops.aten.reshape.default(mul_5, [32, 16, 3, 3]);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_1: "f32[128, 32, 112, 112]" = torch.ops.aten.convolution.default(div, view_5, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  div = view_5 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        neg_1: "f32[128, 32, 112, 112]" = torch.ops.aten.neg.default(convolution_1)
        exp_1: "f32[128, 32, 112, 112]" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_3: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[128, 32, 112, 112]" = torch.ops.aten.div.Tensor(convolution_1, add_3);  convolution_1 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_2: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, getitem_5);  view_6 = getitem_5 = None
        add_4: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_7: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_6: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg8_1, 0.10536653122135592);  arg8_1 = None
        view_7: "f32[64]" = torch.ops.aten.reshape.default(mul_6, [-1]);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_7, -1);  view_7 = None
        mul_8: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_2);  mul_7 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_8: "f32[64, 32, 3, 3]" = torch.ops.aten.reshape.default(mul_8, [64, 32, 3, 3]);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_2: "f32[128, 64, 112, 112]" = torch.ops.aten.convolution.default(div_1, view_8, arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  div_1 = view_8 = arg9_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        neg_2: "f32[128, 64, 112, 112]" = torch.ops.aten.neg.default(convolution_2)
        exp_2: "f32[128, 64, 112, 112]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_5: "f32[128, 64, 112, 112]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[128, 64, 112, 112]" = torch.ops.aten.div.Tensor(convolution_2, add_5);  convolution_2 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_3: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, getitem_7);  view_9 = getitem_7 = None
        add_6: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_10: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_9: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg11_1, 0.07450538873672485);  arg11_1 = None
        view_10: "f32[128]" = torch.ops.aten.reshape.default(mul_9, [-1]);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_3: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_10, -1);  view_10 = None
        mul_11: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_3);  mul_10 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_11: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_11, [128, 64, 3, 3]);  mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_3: "f32[128, 128, 56, 56]" = torch.ops.aten.convolution.default(div_2, view_11, arg12_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div_2 = view_11 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_3: "f32[128, 128, 56, 56]" = torch.ops.aten.neg.default(convolution_3)
        exp_3: "f32[128, 128, 56, 56]" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_7: "f32[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[128, 128, 56, 56]" = torch.ops.aten.div.Tensor(convolution_3, add_7);  convolution_3 = add_7 = None
        mul_12: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(div_3, 1.0);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_5: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_15, getitem_11);  view_15 = getitem_11 = None
        add_9: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_17: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_16: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg17_1, 0.1580497968320339);  arg17_1 = None
        view_16: "f32[64]" = torch.ops.aten.reshape.default(mul_16, [-1]);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_5: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_16, -1);  view_16 = None
        mul_18: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_5);  mul_17 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_17: "f32[64, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_18, [64, 128, 1, 1]);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_5: "f32[128, 64, 56, 56]" = torch.ops.aten.convolution.default(mul_12, view_17, arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_17 = arg18_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_4: "f32[128, 64, 56, 56]" = torch.ops.aten.neg.default(convolution_5)
        exp_4: "f32[128, 64, 56, 56]" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_10: "f32[128, 64, 56, 56]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[128, 64, 56, 56]" = torch.ops.aten.div.Tensor(convolution_5, add_10);  convolution_5 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_6: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_18, getitem_13);  view_18 = getitem_13 = None
        add_11: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_20: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_19: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg20_1, 0.07450538873672485);  arg20_1 = None
        view_19: "f32[64]" = torch.ops.aten.reshape.default(mul_19, [-1]);  mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_19, -1);  view_19 = None
        mul_21: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_6);  mul_20 = unsqueeze_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_20: "f32[64, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_21, [64, 64, 3, 3]);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_6: "f32[128, 64, 56, 56]" = torch.ops.aten.convolution.default(div_4, view_20, arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  div_4 = view_20 = arg21_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_5: "f32[128, 64, 56, 56]" = torch.ops.aten.neg.default(convolution_6)
        exp_5: "f32[128, 64, 56, 56]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_12: "f32[128, 64, 56, 56]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[128, 64, 56, 56]" = torch.ops.aten.div.Tensor(convolution_6, add_12);  convolution_6 = add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_7: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_21, getitem_15);  view_21 = getitem_15 = None
        add_13: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_23: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_22: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg23_1, 0.07450538873672485);  arg23_1 = None
        view_22: "f32[64]" = torch.ops.aten.reshape.default(mul_22, [-1]);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_7: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_22, -1);  view_22 = None
        mul_24: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_23, unsqueeze_7);  mul_23 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_23: "f32[64, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_24, [64, 64, 3, 3]);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_7: "f32[128, 64, 56, 56]" = torch.ops.aten.convolution.default(div_5, view_23, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  div_5 = view_23 = arg24_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_6: "f32[128, 64, 56, 56]" = torch.ops.aten.neg.default(convolution_7)
        exp_6: "f32[128, 64, 56, 56]" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_14: "f32[128, 64, 56, 56]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[128, 64, 56, 56]" = torch.ops.aten.div.Tensor(convolution_7, add_14);  convolution_7 = add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_8: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_24, getitem_17);  view_24 = getitem_17 = None
        add_15: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_26: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_25: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg26_1, 0.22351616621017456);  arg26_1 = None
        view_25: "f32[256]" = torch.ops.aten.reshape.default(mul_25, [-1]);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_8: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_25, -1);  view_25 = None
        mul_27: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(mul_26, unsqueeze_8);  mul_26 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_26: "f32[256, 64, 1, 1]" = torch.ops.aten.reshape.default(mul_27, [256, 64, 1, 1]);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_8: "f32[128, 256, 56, 56]" = torch.ops.aten.convolution.default(div_6, view_26, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_6 = view_26 = arg27_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean: "f32[128, 256, 1, 1]" = torch.ops.aten.mean.dim(convolution_8, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_9: "f32[128, 64, 1, 1]" = torch.ops.aten.convolution.default(mean, arg28_1, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg28_1 = arg29_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu: "f32[128, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_10: "f32[128, 256, 1, 1]" = torch.ops.aten.convolution.default(relu, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu = arg30_1 = arg31_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid: "f32[128, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_28: "f32[128, 256, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid);  convolution_8 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_29: "f32[128, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, 2.0);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_30: "f32[128, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_29, 0.2);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_4: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, getitem_9);  view_12 = getitem_9 = None
        add_8: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_14: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_13: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg14_1, 0.1580497968320339);  arg14_1 = None
        view_13: "f32[256]" = torch.ops.aten.reshape.default(mul_13, [-1]);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_4: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_13, -1);  view_13 = None
        mul_15: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_4);  mul_14 = unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_14: "f32[256, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_15, [256, 128, 1, 1]);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_4: "f32[128, 256, 56, 56]" = torch.ops.aten.convolution.default(mul_12, view_14, arg15_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_12 = view_14 = arg15_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        add_16: "f32[128, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_30, convolution_4);  mul_30 = convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_7: "f32[128, 256, 56, 56]" = torch.ops.aten.neg.default(add_16)
        exp_7: "f32[128, 256, 56, 56]" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_17: "f32[128, 256, 56, 56]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[128, 256, 56, 56]" = torch.ops.aten.div.Tensor(add_16, add_17);  add_16 = add_17 = None
        mul_31: "f32[128, 256, 56, 56]" = torch.ops.aten.mul.Tensor(div_7, 0.9805806756909201);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_10: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_30, getitem_21);  view_30 = getitem_21 = None
        add_19: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_36: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_35: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg36_1, 0.11175808310508728);  arg36_1 = None
        view_31: "f32[128]" = torch.ops.aten.reshape.default(mul_35, [-1]);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_31, -1);  view_31 = None
        mul_37: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_10);  mul_36 = unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_32: "f32[128, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_37, [128, 256, 1, 1]);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_12: "f32[128, 128, 56, 56]" = torch.ops.aten.convolution.default(mul_31, view_32, arg37_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_32 = arg37_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_8: "f32[128, 128, 56, 56]" = torch.ops.aten.neg.default(convolution_12)
        exp_8: "f32[128, 128, 56, 56]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_20: "f32[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[128, 128, 56, 56]" = torch.ops.aten.div.Tensor(convolution_12, add_20);  convolution_12 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_11: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_33, getitem_23);  view_33 = getitem_23 = None
        add_21: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_39: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_38: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg39_1, 0.07450538873672485);  arg39_1 = None
        view_34: "f32[128]" = torch.ops.aten.reshape.default(mul_38, [-1]);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_11: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_34, -1);  view_34 = None
        mul_40: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_11);  mul_39 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_35: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_40, [128, 64, 3, 3]);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_13: "f32[128, 128, 28, 28]" = torch.ops.aten.convolution.default(div_8, view_35, arg40_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 2);  div_8 = view_35 = arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_9: "f32[128, 128, 28, 28]" = torch.ops.aten.neg.default(convolution_13)
        exp_9: "f32[128, 128, 28, 28]" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_22: "f32[128, 128, 28, 28]" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[128, 128, 28, 28]" = torch.ops.aten.div.Tensor(convolution_13, add_22);  convolution_13 = add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_12: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_36, getitem_25);  view_36 = getitem_25 = None
        add_23: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_42: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_41: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg42_1, 0.07450538873672485);  arg42_1 = None
        view_37: "f32[128]" = torch.ops.aten.reshape.default(mul_41, [-1]);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_37, -1);  view_37 = None
        mul_43: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_12);  mul_42 = unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_38: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_43, [128, 64, 3, 3]);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_14: "f32[128, 128, 28, 28]" = torch.ops.aten.convolution.default(div_9, view_38, arg43_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  div_9 = view_38 = arg43_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_10: "f32[128, 128, 28, 28]" = torch.ops.aten.neg.default(convolution_14)
        exp_10: "f32[128, 128, 28, 28]" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_24: "f32[128, 128, 28, 28]" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[128, 128, 28, 28]" = torch.ops.aten.div.Tensor(convolution_14, add_24);  convolution_14 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_13: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_39, getitem_27);  view_39 = getitem_27 = None
        add_25: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_45: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_44: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg45_1, 0.1580497968320339);  arg45_1 = None
        view_40: "f32[512]" = torch.ops.aten.reshape.default(mul_44, [-1]);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_13: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_40, -1);  view_40 = None
        mul_46: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_45, unsqueeze_13);  mul_45 = unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_41: "f32[512, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_46, [512, 128, 1, 1]);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_15: "f32[128, 512, 28, 28]" = torch.ops.aten.convolution.default(div_10, view_41, arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_10 = view_41 = arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_1: "f32[128, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_15, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_16: "f32[128, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg47_1 = arg48_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_1: "f32[128, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_17: "f32[128, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_1, arg49_1, arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_1 = arg49_1 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_1: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_47: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_1);  convolution_15 = sigmoid_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_48: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_47, 2.0);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_49: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_48, 0.2);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d: "f32[128, 256, 28, 28]" = torch.ops.aten.avg_pool2d.default(mul_31, [2, 2], [2, 2], [0, 0], True, False);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, getitem_19);  view_27 = getitem_19 = None
        add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_33: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_32: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg33_1, 0.11175808310508728);  arg33_1 = None
        view_28: "f32[512]" = torch.ops.aten.reshape.default(mul_32, [-1]);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_9: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_28, -1);  view_28 = None
        mul_34: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_9);  mul_33 = unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_29: "f32[512, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_34, [512, 256, 1, 1]);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_11: "f32[128, 512, 28, 28]" = torch.ops.aten.convolution.default(avg_pool2d, view_29, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d = view_29 = arg34_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        add_26: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_49, convolution_11);  mul_49 = convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_11: "f32[128, 512, 28, 28]" = torch.ops.aten.neg.default(add_26)
        exp_11: "f32[128, 512, 28, 28]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_27: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[128, 512, 28, 28]" = torch.ops.aten.div.Tensor(add_26, add_27);  add_27 = None
        mul_50: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(div_11, 0.9805806756909201);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_14: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_42, getitem_29);  view_42 = getitem_29 = None
        add_28: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_52: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_51: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg52_1, 0.07902489841601695);  arg52_1 = None
        view_43: "f32[128]" = torch.ops.aten.reshape.default(mul_51, [-1]);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_43, -1);  view_43 = None
        mul_53: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_14);  mul_52 = unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_44: "f32[128, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_53, [128, 512, 1, 1]);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_18: "f32[128, 128, 28, 28]" = torch.ops.aten.convolution.default(mul_50, view_44, arg53_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_50 = view_44 = arg53_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_12: "f32[128, 128, 28, 28]" = torch.ops.aten.neg.default(convolution_18)
        exp_12: "f32[128, 128, 28, 28]" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_29: "f32[128, 128, 28, 28]" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[128, 128, 28, 28]" = torch.ops.aten.div.Tensor(convolution_18, add_29);  convolution_18 = add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_15: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_45, getitem_31);  view_45 = getitem_31 = None
        add_30: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_55: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_54: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg55_1, 0.07450538873672485);  arg55_1 = None
        view_46: "f32[128]" = torch.ops.aten.reshape.default(mul_54, [-1]);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_15: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_46, -1);  view_46 = None
        mul_56: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_15);  mul_55 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_47: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_56, [128, 64, 3, 3]);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_19: "f32[128, 128, 28, 28]" = torch.ops.aten.convolution.default(div_12, view_47, arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  div_12 = view_47 = arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_13: "f32[128, 128, 28, 28]" = torch.ops.aten.neg.default(convolution_19)
        exp_13: "f32[128, 128, 28, 28]" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_31: "f32[128, 128, 28, 28]" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[128, 128, 28, 28]" = torch.ops.aten.div.Tensor(convolution_19, add_31);  convolution_19 = add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_16: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_48, getitem_33);  view_48 = getitem_33 = None
        add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_58: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_57: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg58_1, 0.07450538873672485);  arg58_1 = None
        view_49: "f32[128]" = torch.ops.aten.reshape.default(mul_57, [-1]);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_49, -1);  view_49 = None
        mul_59: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_16);  mul_58 = unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_50: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_59, [128, 64, 3, 3]);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_20: "f32[128, 128, 28, 28]" = torch.ops.aten.convolution.default(div_13, view_50, arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  div_13 = view_50 = arg59_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_14: "f32[128, 128, 28, 28]" = torch.ops.aten.neg.default(convolution_20)
        exp_14: "f32[128, 128, 28, 28]" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_33: "f32[128, 128, 28, 28]" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[128, 128, 28, 28]" = torch.ops.aten.div.Tensor(convolution_20, add_33);  convolution_20 = add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_17: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_51, getitem_35);  view_51 = getitem_35 = None
        add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_61: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_60: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg61_1, 0.1580497968320339);  arg61_1 = None
        view_52: "f32[512]" = torch.ops.aten.reshape.default(mul_60, [-1]);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_17: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_52, -1);  view_52 = None
        mul_62: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_17);  mul_61 = unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_53: "f32[512, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_62, [512, 128, 1, 1]);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_21: "f32[128, 512, 28, 28]" = torch.ops.aten.convolution.default(div_14, view_53, arg62_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_14 = view_53 = arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_2: "f32[128, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_21, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_22: "f32[128, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_2, arg63_1, arg64_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg63_1 = arg64_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_2: "f32[128, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_23: "f32[128, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_2, arg65_1, arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg65_1 = arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_2: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_63: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_2);  convolution_21 = sigmoid_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_64: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_63, 2.0);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_65: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, 0.2);  mul_64 = None
        add_35: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_65, add_26);  mul_65 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_15: "f32[128, 512, 28, 28]" = torch.ops.aten.neg.default(add_35)
        exp_15: "f32[128, 512, 28, 28]" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_36: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[128, 512, 28, 28]" = torch.ops.aten.div.Tensor(add_35, add_36);  add_35 = add_36 = None
        mul_66: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(div_15, 0.9622504486493761);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_19: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_57, getitem_39);  view_57 = getitem_39 = None
        add_38: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_71: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_70: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg71_1, 0.07902489841601695);  arg71_1 = None
        view_58: "f32[384]" = torch.ops.aten.reshape.default(mul_70, [-1]);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_19: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_58, -1);  view_58 = None
        mul_72: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_19);  mul_71 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_59: "f32[384, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_72, [384, 512, 1, 1]);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_25: "f32[128, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_66, view_59, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_59 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_16: "f32[128, 384, 28, 28]" = torch.ops.aten.neg.default(convolution_25)
        exp_16: "f32[128, 384, 28, 28]" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_39: "f32[128, 384, 28, 28]" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[128, 384, 28, 28]" = torch.ops.aten.div.Tensor(convolution_25, add_39);  convolution_25 = add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_20: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_60, getitem_41);  view_60 = getitem_41 = None
        add_40: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_74: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_73: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg74_1, 0.07450538873672485);  arg74_1 = None
        view_61: "f32[384]" = torch.ops.aten.reshape.default(mul_73, [-1]);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_20: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_61, -1);  view_61 = None
        mul_75: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_20);  mul_74 = unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_62: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_75, [384, 64, 3, 3]);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_26: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_16, view_62, arg75_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  div_16 = view_62 = arg75_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_17: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_26)
        exp_17: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_41: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_26, add_41);  convolution_26 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_21: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_63, getitem_43);  view_63 = getitem_43 = None
        add_42: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_77: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_76: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg77_1, 0.07450538873672485);  arg77_1 = None
        view_64: "f32[384]" = torch.ops.aten.reshape.default(mul_76, [-1]);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_21: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_64, -1);  view_64 = None
        mul_78: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_21);  mul_77 = unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_65: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_78, [384, 64, 3, 3]);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_27: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_17, view_65, arg78_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_17 = view_65 = arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_18: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_27)
        exp_18: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_43: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_27, add_43);  convolution_27 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_22: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_66, getitem_45);  view_66 = getitem_45 = None
        add_44: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_80: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_79: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg80_1, 0.09125009274634042);  arg80_1 = None
        view_67: "f32[1536]" = torch.ops.aten.reshape.default(mul_79, [-1]);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_22: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_67, -1);  view_67 = None
        mul_81: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_22);  mul_80 = unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_68: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_81, [1536, 384, 1, 1]);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_28: "f32[128, 1536, 14, 14]" = torch.ops.aten.convolution.default(div_18, view_68, arg81_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_18 = view_68 = arg81_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_3: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_28, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_29: "f32[128, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_3, arg82_1, arg83_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg82_1 = arg83_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_3: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_29);  convolution_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_30: "f32[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_3, arg84_1, arg85_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_3 = arg84_1 = arg85_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_3: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_82: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_3);  convolution_28 = sigmoid_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_83: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_82, 2.0);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_84: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_83, 0.2);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_1: "f32[128, 512, 14, 14]" = torch.ops.aten.avg_pool2d.default(mul_66, [2, 2], [2, 2], [0, 0], True, False);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_18: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, getitem_37);  view_54 = getitem_37 = None
        add_37: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_68: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_67: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg68_1, 0.07902489841601695);  arg68_1 = None
        view_55: "f32[1536]" = torch.ops.aten.reshape.default(mul_67, [-1]);  mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_18: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_55, -1);  view_55 = None
        mul_69: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_18);  mul_68 = unsqueeze_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_56: "f32[1536, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_69, [1536, 512, 1, 1]);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_24: "f32[128, 1536, 14, 14]" = torch.ops.aten.convolution.default(avg_pool2d_1, view_56, arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_1 = view_56 = arg69_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        add_45: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_84, convolution_24);  mul_84 = convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_19: "f32[128, 1536, 14, 14]" = torch.ops.aten.neg.default(add_45)
        exp_19: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_46: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[128, 1536, 14, 14]" = torch.ops.aten.div.Tensor(add_45, add_46);  add_46 = None
        mul_85: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(div_19, 0.9805806756909201);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_23: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_69, getitem_47);  view_69 = getitem_47 = None
        add_47: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_87: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_86: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg87_1, 0.04562504637317021);  arg87_1 = None
        view_70: "f32[384]" = torch.ops.aten.reshape.default(mul_86, [-1]);  mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_23: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_70, -1);  view_70 = None
        mul_88: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_23);  mul_87 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_71: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_88, [384, 1536, 1, 1]);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_31: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_85, view_71, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_85 = view_71 = arg88_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_20: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_31)
        exp_20: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_48: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_31, add_48);  convolution_31 = add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_24: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_72, getitem_49);  view_72 = getitem_49 = None
        add_49: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_90: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_89: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg90_1, 0.07450538873672485);  arg90_1 = None
        view_73: "f32[384]" = torch.ops.aten.reshape.default(mul_89, [-1]);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_24: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_73, -1);  view_73 = None
        mul_91: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_90, unsqueeze_24);  mul_90 = unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_74: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_91, [384, 64, 3, 3]);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_32: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_20, view_74, arg91_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_20 = view_74 = arg91_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_21: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_32)
        exp_21: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_50: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_32, add_50);  convolution_32 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_25: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_75, getitem_51);  view_75 = getitem_51 = None
        add_51: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_93: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_92: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg93_1, 0.07450538873672485);  arg93_1 = None
        view_76: "f32[384]" = torch.ops.aten.reshape.default(mul_92, [-1]);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_25: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_76, -1);  view_76 = None
        mul_94: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_25);  mul_93 = unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_77: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_94, [384, 64, 3, 3]);  mul_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_33: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_21, view_77, arg94_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_21 = view_77 = arg94_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_22: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_33)
        exp_22: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_52: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_33, add_52);  convolution_33 = add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_26: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_78, getitem_53);  view_78 = getitem_53 = None
        add_53: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_96: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_95: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg96_1, 0.09125009274634042);  arg96_1 = None
        view_79: "f32[1536]" = torch.ops.aten.reshape.default(mul_95, [-1]);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_26: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_79, -1);  view_79 = None
        mul_97: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_26);  mul_96 = unsqueeze_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_80: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_97, [1536, 384, 1, 1]);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_34: "f32[128, 1536, 14, 14]" = torch.ops.aten.convolution.default(div_22, view_80, arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_22 = view_80 = arg97_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_4: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_34, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_35: "f32[128, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_4, arg98_1, arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg98_1 = arg99_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_4: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_36: "f32[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_4, arg100_1, arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg100_1 = arg101_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_4: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_98: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, sigmoid_4);  convolution_34 = sigmoid_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_99: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_98, 2.0);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_100: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_99, 0.2);  mul_99 = None
        add_54: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_100, add_45);  mul_100 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_23: "f32[128, 1536, 14, 14]" = torch.ops.aten.neg.default(add_54)
        exp_23: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_55: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[128, 1536, 14, 14]" = torch.ops.aten.div.Tensor(add_54, add_55);  add_55 = None
        mul_101: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(div_23, 0.9622504486493761);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_27: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_81, getitem_55);  view_81 = getitem_55 = None
        add_56: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_103: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_102: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg103_1, 0.04562504637317021);  arg103_1 = None
        view_82: "f32[384]" = torch.ops.aten.reshape.default(mul_102, [-1]);  mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_27: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_82, -1);  view_82 = None
        mul_104: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_27);  mul_103 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_83: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_104, [384, 1536, 1, 1]);  mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_37: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_101, view_83, arg104_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_101 = view_83 = arg104_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_24: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_37)
        exp_24: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_57: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_37, add_57);  convolution_37 = add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_28: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_84, getitem_57);  view_84 = getitem_57 = None
        add_58: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_106: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_105: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg106_1, 0.07450538873672485);  arg106_1 = None
        view_85: "f32[384]" = torch.ops.aten.reshape.default(mul_105, [-1]);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_28: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_85, -1);  view_85 = None
        mul_107: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_28);  mul_106 = unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_86: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_107, [384, 64, 3, 3]);  mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_38: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_24, view_86, arg107_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_24 = view_86 = arg107_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_25: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_38)
        exp_25: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_59: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_38, add_59);  convolution_38 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_29: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_87, getitem_59);  view_87 = getitem_59 = None
        add_60: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_109: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_108: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg109_1, 0.07450538873672485);  arg109_1 = None
        view_88: "f32[384]" = torch.ops.aten.reshape.default(mul_108, [-1]);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_29: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_88, -1);  view_88 = None
        mul_110: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_29);  mul_109 = unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_89: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_110, [384, 64, 3, 3]);  mul_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_39: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_25, view_89, arg110_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_25 = view_89 = arg110_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_26: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_39)
        exp_26: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_61: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_39, add_61);  convolution_39 = add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_30: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_90, getitem_61);  view_90 = getitem_61 = None
        add_62: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_112: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_111: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg112_1, 0.09125009274634042);  arg112_1 = None
        view_91: "f32[1536]" = torch.ops.aten.reshape.default(mul_111, [-1]);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_30: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_91, -1);  view_91 = None
        mul_113: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_30);  mul_112 = unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_92: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_113, [1536, 384, 1, 1]);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_40: "f32[128, 1536, 14, 14]" = torch.ops.aten.convolution.default(div_26, view_92, arg113_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_26 = view_92 = arg113_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_5: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_40, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_41: "f32[128, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_5, arg114_1, arg115_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg114_1 = arg115_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_5: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_41);  convolution_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_42: "f32[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_5, arg116_1, arg117_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg116_1 = arg117_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_5: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_114: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_40, sigmoid_5);  convolution_40 = sigmoid_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_115: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_114, 2.0);  mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_116: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, 0.2);  mul_115 = None
        add_63: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, add_54);  mul_116 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_27: "f32[128, 1536, 14, 14]" = torch.ops.aten.neg.default(add_63)
        exp_27: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_64: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[128, 1536, 14, 14]" = torch.ops.aten.div.Tensor(add_63, add_64);  add_64 = None
        mul_117: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(div_27, 0.9449111825230679);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_31: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_93, getitem_63);  view_93 = getitem_63 = None
        add_65: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_119: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_118: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg119_1, 0.04562504637317021);  arg119_1 = None
        view_94: "f32[384]" = torch.ops.aten.reshape.default(mul_118, [-1]);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_31: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_94, -1);  view_94 = None
        mul_120: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_31);  mul_119 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_95: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_120, [384, 1536, 1, 1]);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_43: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_117, view_95, arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_117 = view_95 = arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_28: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_43)
        exp_28: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_66: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_43, add_66);  convolution_43 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_32: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_96, getitem_65);  view_96 = getitem_65 = None
        add_67: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_122: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_121: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg122_1, 0.07450538873672485);  arg122_1 = None
        view_97: "f32[384]" = torch.ops.aten.reshape.default(mul_121, [-1]);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_32: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_97, -1);  view_97 = None
        mul_123: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_122, unsqueeze_32);  mul_122 = unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_98: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_123, [384, 64, 3, 3]);  mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_44: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_28, view_98, arg123_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_28 = view_98 = arg123_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_29: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_44)
        exp_29: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_68: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        div_29: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_44, add_68);  convolution_44 = add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_33: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_99, getitem_67);  view_99 = getitem_67 = None
        add_69: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_125: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_124: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg125_1, 0.07450538873672485);  arg125_1 = None
        view_100: "f32[384]" = torch.ops.aten.reshape.default(mul_124, [-1]);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_33: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_100, -1);  view_100 = None
        mul_126: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_125, unsqueeze_33);  mul_125 = unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_101: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_126, [384, 64, 3, 3]);  mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_45: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_29, view_101, arg126_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_29 = view_101 = arg126_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_30: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_45)
        exp_30: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_70: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        div_30: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_45, add_70);  convolution_45 = add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_34: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_102, getitem_69);  view_102 = getitem_69 = None
        add_71: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_128: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_127: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg128_1, 0.09125009274634042);  arg128_1 = None
        view_103: "f32[1536]" = torch.ops.aten.reshape.default(mul_127, [-1]);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_34: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_103, -1);  view_103 = None
        mul_129: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_34);  mul_128 = unsqueeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_104: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_129, [1536, 384, 1, 1]);  mul_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_46: "f32[128, 1536, 14, 14]" = torch.ops.aten.convolution.default(div_30, view_104, arg129_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_30 = view_104 = arg129_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_6: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_46, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_47: "f32[128, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_6, arg130_1, arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg130_1 = arg131_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_6: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_48: "f32[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_6, arg132_1, arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_6 = arg132_1 = arg133_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_6: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_130: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_46, sigmoid_6);  convolution_46 = sigmoid_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_131: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, 2.0);  mul_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_132: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_131, 0.2);  mul_131 = None
        add_72: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_132, add_63);  mul_132 = add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_31: "f32[128, 1536, 14, 14]" = torch.ops.aten.neg.default(add_72)
        exp_31: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_73: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[128, 1536, 14, 14]" = torch.ops.aten.div.Tensor(add_72, add_73);  add_73 = None
        mul_133: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(div_31, 0.9284766908852592);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_35: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_105, getitem_71);  view_105 = getitem_71 = None
        add_74: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_135: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_134: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg135_1, 0.04562504637317021);  arg135_1 = None
        view_106: "f32[384]" = torch.ops.aten.reshape.default(mul_134, [-1]);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_35: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_106, -1);  view_106 = None
        mul_136: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_35);  mul_135 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_107: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_136, [384, 1536, 1, 1]);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_49: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_133, view_107, arg136_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_133 = view_107 = arg136_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_32: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_49)
        exp_32: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_75: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        div_32: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_49, add_75);  convolution_49 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_36: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_108, getitem_73);  view_108 = getitem_73 = None
        add_76: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_138: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_137: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg138_1, 0.07450538873672485);  arg138_1 = None
        view_109: "f32[384]" = torch.ops.aten.reshape.default(mul_137, [-1]);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_36: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_109, -1);  view_109 = None
        mul_139: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_36);  mul_138 = unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_110: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_139, [384, 64, 3, 3]);  mul_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_50: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_32, view_110, arg139_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_32 = view_110 = arg139_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_33: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_50)
        exp_33: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_77: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        div_33: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_50, add_77);  convolution_50 = add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_37: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_111, getitem_75);  view_111 = getitem_75 = None
        add_78: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_141: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_140: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg141_1, 0.07450538873672485);  arg141_1 = None
        view_112: "f32[384]" = torch.ops.aten.reshape.default(mul_140, [-1]);  mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_37: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_112, -1);  view_112 = None
        mul_142: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_37);  mul_141 = unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_113: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_142, [384, 64, 3, 3]);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_51: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_33, view_113, arg142_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_33 = view_113 = arg142_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_34: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_51)
        exp_34: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_34);  neg_34 = None
        add_79: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_34, 1);  exp_34 = None
        div_34: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_51, add_79);  convolution_51 = add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_38: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_114, getitem_77);  view_114 = getitem_77 = None
        add_80: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_144: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_143: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg144_1, 0.09125009274634042);  arg144_1 = None
        view_115: "f32[1536]" = torch.ops.aten.reshape.default(mul_143, [-1]);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_38: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_115, -1);  view_115 = None
        mul_145: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_38);  mul_144 = unsqueeze_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_116: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_145, [1536, 384, 1, 1]);  mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_52: "f32[128, 1536, 14, 14]" = torch.ops.aten.convolution.default(div_34, view_116, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_34 = view_116 = arg145_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_7: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_52, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_53: "f32[128, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_7, arg146_1, arg147_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg146_1 = arg147_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_7: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_54: "f32[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_7, arg148_1, arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_7 = arg148_1 = arg149_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_7: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_54);  convolution_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_146: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_7);  convolution_52 = sigmoid_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_147: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_146, 2.0);  mul_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_148: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_147, 0.2);  mul_147 = None
        add_81: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_148, add_72);  mul_148 = add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_35: "f32[128, 1536, 14, 14]" = torch.ops.aten.neg.default(add_81)
        exp_35: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_82: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(exp_35, 1);  exp_35 = None
        div_35: "f32[128, 1536, 14, 14]" = torch.ops.aten.div.Tensor(add_81, add_82);  add_82 = None
        mul_149: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(div_35, 0.9128709291752768);  div_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_39: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_117, getitem_79);  view_117 = getitem_79 = None
        add_83: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_151: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_150: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg151_1, 0.04562504637317021);  arg151_1 = None
        view_118: "f32[384]" = torch.ops.aten.reshape.default(mul_150, [-1]);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_39: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_118, -1);  view_118 = None
        mul_152: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_39);  mul_151 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_119: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_152, [384, 1536, 1, 1]);  mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_55: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_149, view_119, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_149 = view_119 = arg152_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_36: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_55)
        exp_36: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_36);  neg_36 = None
        add_84: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_36, 1);  exp_36 = None
        div_36: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_55, add_84);  convolution_55 = add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_40: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_120, getitem_81);  view_120 = getitem_81 = None
        add_85: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_154: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_153: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg154_1, 0.07450538873672485);  arg154_1 = None
        view_121: "f32[384]" = torch.ops.aten.reshape.default(mul_153, [-1]);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_40: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_121, -1);  view_121 = None
        mul_155: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_40);  mul_154 = unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_122: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_155, [384, 64, 3, 3]);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_56: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_36, view_122, arg155_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_36 = view_122 = arg155_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_37: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_56)
        exp_37: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_37);  neg_37 = None
        add_86: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_37, 1);  exp_37 = None
        div_37: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_56, add_86);  convolution_56 = add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_41: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_123, getitem_83);  view_123 = getitem_83 = None
        add_87: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_157: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_156: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg157_1, 0.07450538873672485);  arg157_1 = None
        view_124: "f32[384]" = torch.ops.aten.reshape.default(mul_156, [-1]);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_41: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_124, -1);  view_124 = None
        mul_158: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_41);  mul_157 = unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_125: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_158, [384, 64, 3, 3]);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_57: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(div_37, view_125, arg158_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_37 = view_125 = arg158_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_38: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_57)
        exp_38: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_88: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_38, 1);  exp_38 = None
        div_38: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_57, add_88);  convolution_57 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_42: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_126, getitem_85);  view_126 = getitem_85 = None
        add_89: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_160: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_159: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg160_1, 0.09125009274634042);  arg160_1 = None
        view_127: "f32[1536]" = torch.ops.aten.reshape.default(mul_159, [-1]);  mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_42: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_127, -1);  view_127 = None
        mul_161: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_42);  mul_160 = unsqueeze_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_128: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_161, [1536, 384, 1, 1]);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_58: "f32[128, 1536, 14, 14]" = torch.ops.aten.convolution.default(div_38, view_128, arg161_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_38 = view_128 = arg161_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_8: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_58, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_59: "f32[128, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_8, arg162_1, arg163_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg162_1 = arg163_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_8: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_59);  convolution_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_60: "f32[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_8, arg164_1, arg165_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg164_1 = arg165_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_8: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_162: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_8);  convolution_58 = sigmoid_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_163: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_162, 2.0);  mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_164: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_163, 0.2);  mul_163 = None
        add_90: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_164, add_81);  mul_164 = add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_39: "f32[128, 1536, 14, 14]" = torch.ops.aten.neg.default(add_90)
        exp_39: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(neg_39);  neg_39 = None
        add_91: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(exp_39, 1);  exp_39 = None
        div_39: "f32[128, 1536, 14, 14]" = torch.ops.aten.div.Tensor(add_90, add_91);  add_90 = add_91 = None
        mul_165: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(div_39, 0.8980265101338745);  div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_44: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_132, getitem_89);  view_132 = getitem_89 = None
        add_93: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_170: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_169: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg170_1, 0.04562504637317021);  arg170_1 = None
        view_133: "f32[384]" = torch.ops.aten.reshape.default(mul_169, [-1]);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_44: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_133, -1);  view_133 = None
        mul_171: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_44);  mul_170 = unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_134: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_171, [384, 1536, 1, 1]);  mul_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_62: "f32[128, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_165, view_134, arg171_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_134 = arg171_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_40: "f32[128, 384, 14, 14]" = torch.ops.aten.neg.default(convolution_62)
        exp_40: "f32[128, 384, 14, 14]" = torch.ops.aten.exp.default(neg_40);  neg_40 = None
        add_94: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(exp_40, 1);  exp_40 = None
        div_40: "f32[128, 384, 14, 14]" = torch.ops.aten.div.Tensor(convolution_62, add_94);  convolution_62 = add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_45: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_135, getitem_91);  view_135 = getitem_91 = None
        add_95: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_173: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_172: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg173_1, 0.07450538873672485);  arg173_1 = None
        view_136: "f32[384]" = torch.ops.aten.reshape.default(mul_172, [-1]);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_45: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_136, -1);  view_136 = None
        mul_174: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_45);  mul_173 = unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_137: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_174, [384, 64, 3, 3]);  mul_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_63: "f32[128, 384, 7, 7]" = torch.ops.aten.convolution.default(div_40, view_137, arg174_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  div_40 = view_137 = arg174_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_41: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_63)
        exp_41: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_96: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_41, 1);  exp_41 = None
        div_41: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_63, add_96);  convolution_63 = add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_46: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_138, getitem_93);  view_138 = getitem_93 = None
        add_97: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_176: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_175: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg176_1, 0.07450538873672485);  arg176_1 = None
        view_139: "f32[384]" = torch.ops.aten.reshape.default(mul_175, [-1]);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_46: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_139, -1);  view_139 = None
        mul_177: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_46);  mul_176 = unsqueeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_140: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_177, [384, 64, 3, 3]);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_64: "f32[128, 384, 7, 7]" = torch.ops.aten.convolution.default(div_41, view_140, arg177_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_41 = view_140 = arg177_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_42: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_64)
        exp_42: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_42);  neg_42 = None
        add_98: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_42, 1);  exp_42 = None
        div_42: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_64, add_98);  convolution_64 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_47: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_141, getitem_95);  view_141 = getitem_95 = None
        add_99: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_179: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_178: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg179_1, 0.09125009274634042);  arg179_1 = None
        view_142: "f32[1536]" = torch.ops.aten.reshape.default(mul_178, [-1]);  mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_47: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_142, -1);  view_142 = None
        mul_180: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_179, unsqueeze_47);  mul_179 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_143: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_180, [1536, 384, 1, 1]);  mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_65: "f32[128, 1536, 7, 7]" = torch.ops.aten.convolution.default(div_42, view_143, arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_42 = view_143 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_9: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_65, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_66: "f32[128, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_9, arg181_1, arg182_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg181_1 = arg182_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_9: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_67: "f32[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_9, arg183_1, arg184_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg183_1 = arg184_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_9: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_181: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_9);  convolution_65 = sigmoid_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_182: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_181, 2.0);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_183: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_182, 0.2);  mul_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_2: "f32[128, 1536, 7, 7]" = torch.ops.aten.avg_pool2d.default(mul_165, [2, 2], [2, 2], [0, 0], True, False);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_43: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, getitem_87);  view_129 = getitem_87 = None
        add_92: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_167: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_166: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg167_1, 0.04562504637317021);  arg167_1 = None
        view_130: "f32[1536]" = torch.ops.aten.reshape.default(mul_166, [-1]);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_43: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_130, -1);  view_130 = None
        mul_168: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_43);  mul_167 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_131: "f32[1536, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_168, [1536, 1536, 1, 1]);  mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_61: "f32[128, 1536, 7, 7]" = torch.ops.aten.convolution.default(avg_pool2d_2, view_131, arg168_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_2 = view_131 = arg168_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        add_100: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_183, convolution_61);  mul_183 = convolution_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_43: "f32[128, 1536, 7, 7]" = torch.ops.aten.neg.default(add_100)
        exp_43: "f32[128, 1536, 7, 7]" = torch.ops.aten.exp.default(neg_43);  neg_43 = None
        add_101: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(exp_43, 1);  exp_43 = None
        div_43: "f32[128, 1536, 7, 7]" = torch.ops.aten.div.Tensor(add_100, add_101);  add_101 = None
        mul_184: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(div_43, 0.9805806756909201);  div_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_48: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_144, getitem_97);  view_144 = getitem_97 = None
        add_102: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_186: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_185: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg186_1, 0.04562504637317021);  arg186_1 = None
        view_145: "f32[384]" = torch.ops.aten.reshape.default(mul_185, [-1]);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_48: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_145, -1);  view_145 = None
        mul_187: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_186, unsqueeze_48);  mul_186 = unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_146: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_187, [384, 1536, 1, 1]);  mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_68: "f32[128, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_184, view_146, arg187_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_184 = view_146 = arg187_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_44: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_68)
        exp_44: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_103: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_44, 1);  exp_44 = None
        div_44: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_68, add_103);  convolution_68 = add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_49: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_147, getitem_99);  view_147 = getitem_99 = None
        add_104: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_49: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_189: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_188: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg189_1, 0.07450538873672485);  arg189_1 = None
        view_148: "f32[384]" = torch.ops.aten.reshape.default(mul_188, [-1]);  mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_49: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_148, -1);  view_148 = None
        mul_190: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_49);  mul_189 = unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_149: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_190, [384, 64, 3, 3]);  mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_69: "f32[128, 384, 7, 7]" = torch.ops.aten.convolution.default(div_44, view_149, arg190_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_44 = view_149 = arg190_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_45: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_69)
        exp_45: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_45);  neg_45 = None
        add_105: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_45, 1);  exp_45 = None
        div_45: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_69, add_105);  convolution_69 = add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_50: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_150, getitem_101);  view_150 = getitem_101 = None
        add_106: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_50: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_192: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_191: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg192_1, 0.07450538873672485);  arg192_1 = None
        view_151: "f32[384]" = torch.ops.aten.reshape.default(mul_191, [-1]);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_50: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_151, -1);  view_151 = None
        mul_193: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_50);  mul_192 = unsqueeze_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_152: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_193, [384, 64, 3, 3]);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_70: "f32[128, 384, 7, 7]" = torch.ops.aten.convolution.default(div_45, view_152, arg193_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_45 = view_152 = arg193_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_46: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_70)
        exp_46: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_46);  neg_46 = None
        add_107: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_46, 1);  exp_46 = None
        div_46: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_70, add_107);  convolution_70 = add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_51: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_153, getitem_103);  view_153 = getitem_103 = None
        add_108: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_51: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_195: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_194: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg195_1, 0.09125009274634042);  arg195_1 = None
        view_154: "f32[1536]" = torch.ops.aten.reshape.default(mul_194, [-1]);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_51: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_154, -1);  view_154 = None
        mul_196: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_51);  mul_195 = unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_155: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_196, [1536, 384, 1, 1]);  mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_71: "f32[128, 1536, 7, 7]" = torch.ops.aten.convolution.default(div_46, view_155, arg196_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_46 = view_155 = arg196_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_10: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_71, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_72: "f32[128, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_10, arg197_1, arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg197_1 = arg198_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_10: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_72);  convolution_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_73: "f32[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_10, arg199_1, arg200_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_10 = arg199_1 = arg200_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_10: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_197: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_10);  convolution_71 = sigmoid_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_198: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_197, 2.0);  mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_199: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_198, 0.2);  mul_198 = None
        add_109: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_199, add_100);  mul_199 = add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_47: "f32[128, 1536, 7, 7]" = torch.ops.aten.neg.default(add_109)
        exp_47: "f32[128, 1536, 7, 7]" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_110: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(exp_47, 1);  exp_47 = None
        div_47: "f32[128, 1536, 7, 7]" = torch.ops.aten.div.Tensor(add_109, add_110);  add_110 = None
        mul_200: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(div_47, 0.9622504486493761);  div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_52: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_156, getitem_105);  view_156 = getitem_105 = None
        add_111: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_52: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_202: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_201: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg202_1, 0.04562504637317021);  arg202_1 = None
        view_157: "f32[384]" = torch.ops.aten.reshape.default(mul_201, [-1]);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_52: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_157, -1);  view_157 = None
        mul_203: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_52);  mul_202 = unsqueeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_158: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_203, [384, 1536, 1, 1]);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_74: "f32[128, 384, 7, 7]" = torch.ops.aten.convolution.default(mul_200, view_158, arg203_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_200 = view_158 = arg203_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_48: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_74)
        exp_48: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_48);  neg_48 = None
        add_112: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_48, 1);  exp_48 = None
        div_48: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_74, add_112);  convolution_74 = add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_53: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_159, getitem_107);  view_159 = getitem_107 = None
        add_113: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
        rsqrt_53: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_205: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_204: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg205_1, 0.07450538873672485);  arg205_1 = None
        view_160: "f32[384]" = torch.ops.aten.reshape.default(mul_204, [-1]);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_53: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_160, -1);  view_160 = None
        mul_206: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_53);  mul_205 = unsqueeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_161: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_206, [384, 64, 3, 3]);  mul_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_75: "f32[128, 384, 7, 7]" = torch.ops.aten.convolution.default(div_48, view_161, arg206_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_48 = view_161 = arg206_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_49: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_75)
        exp_49: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_49);  neg_49 = None
        add_114: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_49, 1);  exp_49 = None
        div_49: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_75, add_114);  convolution_75 = add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_54: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_162, getitem_109);  view_162 = getitem_109 = None
        add_115: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_54: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_208: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_207: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg208_1, 0.07450538873672485);  arg208_1 = None
        view_163: "f32[384]" = torch.ops.aten.reshape.default(mul_207, [-1]);  mul_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_54: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_163, -1);  view_163 = None
        mul_209: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_54);  mul_208 = unsqueeze_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_164: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_209, [384, 64, 3, 3]);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_76: "f32[128, 384, 7, 7]" = torch.ops.aten.convolution.default(div_49, view_164, arg209_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  div_49 = view_164 = arg209_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_50: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_76)
        exp_50: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_50);  neg_50 = None
        add_116: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_50, 1);  exp_50 = None
        div_50: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_76, add_116);  convolution_76 = add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_55: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_165, getitem_111);  view_165 = getitem_111 = None
        add_117: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_55: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        mul_211: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_210: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg211_1, 0.09125009274634042);  arg211_1 = None
        view_166: "f32[1536]" = torch.ops.aten.reshape.default(mul_210, [-1]);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_55: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_166, -1);  view_166 = None
        mul_212: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_55);  mul_211 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_167: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_212, [1536, 384, 1, 1]);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_77: "f32[128, 1536, 7, 7]" = torch.ops.aten.convolution.default(div_50, view_167, arg212_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_50 = view_167 = arg212_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_11: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_77, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_78: "f32[128, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_11, arg213_1, arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg213_1 = arg214_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_11: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_78);  convolution_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_79: "f32[128, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_11, arg215_1, arg216_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg215_1 = arg216_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_11: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_213: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_11);  convolution_77 = sigmoid_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_214: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_213, 2.0);  mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_215: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_214, 0.2);  mul_214 = None
        add_118: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_215, add_109);  mul_215 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_56: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_168, getitem_113);  view_168 = getitem_113 = None
        add_119: "f32[1, 2304, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_56: "f32[1, 2304, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_217: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_216: "f32[2304, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg218_1, 0.04562504637317021);  arg218_1 = None
        view_169: "f32[2304]" = torch.ops.aten.reshape.default(mul_216, [-1]);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_56: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(view_169, -1);  view_169 = None
        mul_218: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_56);  mul_217 = unsqueeze_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_170: "f32[2304, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_218, [2304, 1536, 1, 1]);  mul_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_80: "f32[128, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_118, view_170, arg219_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_118 = view_170 = arg219_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:569 in forward_features, code: x = self.final_act(x)
        neg_51: "f32[128, 2304, 7, 7]" = torch.ops.aten.neg.default(convolution_80)
        exp_51: "f32[128, 2304, 7, 7]" = torch.ops.aten.exp.default(neg_51);  neg_51 = None
        add_120: "f32[128, 2304, 7, 7]" = torch.ops.aten.add.Tensor(exp_51, 1);  exp_51 = None
        div_51: "f32[128, 2304, 7, 7]" = torch.ops.aten.div.Tensor(convolution_80, add_120);  convolution_80 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_12: "f32[128, 2304, 1, 1]" = torch.ops.aten.mean.dim(div_51, [-1, -2], True);  div_51 = None
        as_strided: "f32[128, 2304, 1, 1]" = torch.ops.aten.as_strided.default(mean_12, [128, 2304, 1, 1], [2304, 1, 2304, 2304]);  mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_171: "f32[128, 2304]" = torch.ops.aten.reshape.default(as_strided, [128, 2304]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute: "f32[2304, 1000]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm: "f32[128, 1000]" = torch.ops.aten.addmm.default(arg221_1, view_171, permute);  arg221_1 = view_171 = permute = None
        return (addmm,)

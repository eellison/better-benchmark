class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[80, 3, 4, 4][48, 1, 12, 3]cuda:0", arg1_1: "bf16[80][1]cuda:0", arg2_1: "bf16[128, 3, 288, 288][248832, 1, 864, 3]cuda:0", arg3_1: "bf16[80][1]cuda:0", arg4_1: "bf16[80][1]cuda:0", arg5_1: "bf16[80, 1, 7, 7][49, 1, 7, 1]cuda:0", arg6_1: "bf16[80][1]cuda:0", arg7_1: "bf16[80][1]cuda:0", arg8_1: "bf16[80][1]cuda:0", arg9_1: "bf16[320, 80, 1, 1][80, 1, 80, 80]cuda:0", arg10_1: "bf16[320][1]cuda:0", arg11_1: "bf16[320][1]cuda:0", arg12_1: "bf16[320][1]cuda:0", arg13_1: "bf16[80, 320, 1, 1][320, 1, 320, 320]cuda:0", arg14_1: "bf16[80][1]cuda:0", arg15_1: "bf16[80, 1, 7, 7][49, 1, 7, 1]cuda:0", arg16_1: "bf16[80][1]cuda:0", arg17_1: "bf16[80][1]cuda:0", arg18_1: "bf16[80][1]cuda:0", arg19_1: "bf16[320, 80, 1, 1][80, 1, 80, 80]cuda:0", arg20_1: "bf16[320][1]cuda:0", arg21_1: "bf16[320][1]cuda:0", arg22_1: "bf16[320][1]cuda:0", arg23_1: "bf16[80, 320, 1, 1][320, 1, 320, 320]cuda:0", arg24_1: "bf16[80][1]cuda:0", arg25_1: "bf16[80][1]cuda:0", arg26_1: "bf16[80][1]cuda:0", arg27_1: "bf16[160, 80, 2, 2][320, 1, 160, 80]cuda:0", arg28_1: "bf16[160][1]cuda:0", arg29_1: "bf16[160, 1, 7, 7][49, 1, 7, 1]cuda:0", arg30_1: "bf16[160][1]cuda:0", arg31_1: "bf16[160][1]cuda:0", arg32_1: "bf16[160][1]cuda:0", arg33_1: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0", arg34_1: "bf16[640][1]cuda:0", arg35_1: "bf16[640][1]cuda:0", arg36_1: "bf16[640][1]cuda:0", arg37_1: "bf16[160, 640, 1, 1][640, 1, 640, 640]cuda:0", arg38_1: "bf16[160][1]cuda:0", arg39_1: "bf16[160, 1, 7, 7][49, 1, 7, 1]cuda:0", arg40_1: "bf16[160][1]cuda:0", arg41_1: "bf16[160][1]cuda:0", arg42_1: "bf16[160][1]cuda:0", arg43_1: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0", arg44_1: "bf16[640][1]cuda:0", arg45_1: "bf16[640][1]cuda:0", arg46_1: "bf16[640][1]cuda:0", arg47_1: "bf16[160, 640, 1, 1][640, 1, 640, 640]cuda:0", arg48_1: "bf16[160][1]cuda:0", arg49_1: "bf16[160][1]cuda:0", arg50_1: "bf16[160][1]cuda:0", arg51_1: "bf16[320, 160, 2, 2][640, 1, 320, 160]cuda:0", arg52_1: "bf16[320][1]cuda:0", arg53_1: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg54_1: "bf16[320][1]cuda:0", arg55_1: "bf16[320][1]cuda:0", arg56_1: "bf16[320][1]cuda:0", arg57_1: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg58_1: "bf16[1280][1]cuda:0", arg59_1: "bf16[1280][1]cuda:0", arg60_1: "bf16[1280][1]cuda:0", arg61_1: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg62_1: "bf16[320][1]cuda:0", arg63_1: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg64_1: "bf16[320][1]cuda:0", arg65_1: "bf16[320][1]cuda:0", arg66_1: "bf16[320][1]cuda:0", arg67_1: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg68_1: "bf16[1280][1]cuda:0", arg69_1: "bf16[1280][1]cuda:0", arg70_1: "bf16[1280][1]cuda:0", arg71_1: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg72_1: "bf16[320][1]cuda:0", arg73_1: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg74_1: "bf16[320][1]cuda:0", arg75_1: "bf16[320][1]cuda:0", arg76_1: "bf16[320][1]cuda:0", arg77_1: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg78_1: "bf16[1280][1]cuda:0", arg79_1: "bf16[1280][1]cuda:0", arg80_1: "bf16[1280][1]cuda:0", arg81_1: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg82_1: "bf16[320][1]cuda:0", arg83_1: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg84_1: "bf16[320][1]cuda:0", arg85_1: "bf16[320][1]cuda:0", arg86_1: "bf16[320][1]cuda:0", arg87_1: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg88_1: "bf16[1280][1]cuda:0", arg89_1: "bf16[1280][1]cuda:0", arg90_1: "bf16[1280][1]cuda:0", arg91_1: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg92_1: "bf16[320][1]cuda:0", arg93_1: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg94_1: "bf16[320][1]cuda:0", arg95_1: "bf16[320][1]cuda:0", arg96_1: "bf16[320][1]cuda:0", arg97_1: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg98_1: "bf16[1280][1]cuda:0", arg99_1: "bf16[1280][1]cuda:0", arg100_1: "bf16[1280][1]cuda:0", arg101_1: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg102_1: "bf16[320][1]cuda:0", arg103_1: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg104_1: "bf16[320][1]cuda:0", arg105_1: "bf16[320][1]cuda:0", arg106_1: "bf16[320][1]cuda:0", arg107_1: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg108_1: "bf16[1280][1]cuda:0", arg109_1: "bf16[1280][1]cuda:0", arg110_1: "bf16[1280][1]cuda:0", arg111_1: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg112_1: "bf16[320][1]cuda:0", arg113_1: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg114_1: "bf16[320][1]cuda:0", arg115_1: "bf16[320][1]cuda:0", arg116_1: "bf16[320][1]cuda:0", arg117_1: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg118_1: "bf16[1280][1]cuda:0", arg119_1: "bf16[1280][1]cuda:0", arg120_1: "bf16[1280][1]cuda:0", arg121_1: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg122_1: "bf16[320][1]cuda:0", arg123_1: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg124_1: "bf16[320][1]cuda:0", arg125_1: "bf16[320][1]cuda:0", arg126_1: "bf16[320][1]cuda:0", arg127_1: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg128_1: "bf16[1280][1]cuda:0", arg129_1: "bf16[1280][1]cuda:0", arg130_1: "bf16[1280][1]cuda:0", arg131_1: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg132_1: "bf16[320][1]cuda:0", arg133_1: "bf16[320][1]cuda:0", arg134_1: "bf16[320][1]cuda:0", arg135_1: "bf16[640, 320, 2, 2][1280, 1, 640, 320]cuda:0", arg136_1: "bf16[640][1]cuda:0", arg137_1: "bf16[640, 1, 7, 7][49, 1, 7, 1]cuda:0", arg138_1: "bf16[640][1]cuda:0", arg139_1: "bf16[640][1]cuda:0", arg140_1: "bf16[640][1]cuda:0", arg141_1: "bf16[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", arg142_1: "bf16[2560][1]cuda:0", arg143_1: "bf16[2560][1]cuda:0", arg144_1: "bf16[2560][1]cuda:0", arg145_1: "bf16[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", arg146_1: "bf16[640][1]cuda:0", arg147_1: "bf16[640, 1, 7, 7][49, 1, 7, 1]cuda:0", arg148_1: "bf16[640][1]cuda:0", arg149_1: "bf16[640][1]cuda:0", arg150_1: "bf16[640][1]cuda:0", arg151_1: "bf16[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", arg152_1: "bf16[2560][1]cuda:0", arg153_1: "bf16[2560][1]cuda:0", arg154_1: "bf16[2560][1]cuda:0", arg155_1: "bf16[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", arg156_1: "bf16[640][1]cuda:0", arg157_1: "bf16[640][1]cuda:0", arg158_1: "bf16[640][1]cuda:0", arg159_1: "bf16[1000, 640][640, 1]cuda:0", arg160_1: "bf16[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:608 in forward_features, code: x = self.stem(x)
        convolution: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute: "bf16[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type_1: "bf16[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_1: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [0, 3, 1, 2]);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_1: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.convolution.default(permute_1, arg5_1, arg6_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 80);  arg5_1 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_2: "bf16[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_2: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_2, torch.float32);  permute_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_2, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view: "bf16[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg11_1, [1, -1, 1, 1]);  arg11_1 = None
        view_1: "bf16[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg12_1, [1, -1, 1, 1]);  arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_1: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_3);  convert_element_type_2 = getitem_3 = None
        add_2: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
        add_3: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, arg8_1);  mul_3 = arg8_1 = None
        convert_element_type_3: "bf16[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_3: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.permute.default(convert_element_type_3, [0, 3, 1, 2]);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_2: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.convolution.default(permute_3, arg9_1, arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_3 = arg9_1 = arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_4: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        mul_4: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.5)
        mul_5: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.7071067811865476);  convert_element_type_4 = None
        erf: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
        convert_element_type_5: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_6: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32)
        pow_1: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2);  convert_element_type_6 = None
        sum_1: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_1, [2, 3], True);  pow_1 = None
        pow_2: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        convert_element_type_7: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_2, torch.bfloat16);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_7, [1], True)
        add_5: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        div: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_7, add_5);  convert_element_type_7 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_7: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_5, div);  div = None
        addcmul: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.addcmul.default(view, view_1, mul_7);  view = view_1 = mul_7 = None
        add_6: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_5, addcmul);  convert_element_type_5 = addcmul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_3: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.convolution.default(add_6, arg13_1, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_6 = arg13_1 = arg14_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_7: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.add.Tensor(convolution_3, permute_1);  convolution_3 = permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_4: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.convolution.default(add_7, arg15_1, arg16_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 80);  arg15_1 = arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_4: "bf16[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_8: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [3], correction = 0, keepdim = True)
        getitem_4: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_2: "bf16[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg21_1, [1, -1, 1, 1]);  arg21_1 = None
        view_3: "bf16[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg22_1, [1, -1, 1, 1]);  arg22_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_2: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_5);  convert_element_type_8 = getitem_5 = None
        add_8: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
        rsqrt_2: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_8: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, arg17_1);  mul_8 = arg17_1 = None
        add_9: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, arg18_1);  mul_9 = arg18_1 = None
        convert_element_type_9: "bf16[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_5: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.permute.default(convert_element_type_9, [0, 3, 1, 2]);  convert_element_type_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_5: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.convolution.default(permute_5, arg19_1, arg20_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_5 = arg19_1 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_10: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        mul_10: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.5)
        mul_11: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.7071067811865476);  convert_element_type_10 = None
        erf_1: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_10: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_12: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, add_10);  mul_10 = add_10 = None
        convert_element_type_11: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_12: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32)
        pow_3: "f32[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_12, 2);  convert_element_type_12 = None
        sum_2: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_3, [2, 3], True);  pow_3 = None
        pow_4: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None
        convert_element_type_13: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_4, torch.bfloat16);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_1: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_13, [1], True)
        add_11: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        div_1: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_13, add_11);  convert_element_type_13 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_13: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, div_1);  div_1 = None
        addcmul_1: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.addcmul.default(view_2, view_3, mul_13);  view_2 = view_3 = mul_13 = None
        add_12: "bf16[128, 320, 72, 72][1658880, 1, 23040, 320]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_11, addcmul_1);  convert_element_type_11 = addcmul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_6: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.convolution.default(add_12, arg23_1, arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_12 = arg23_1 = arg24_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_13: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.add.Tensor(convolution_6, add_7);  convolution_6 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_6: "bf16[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.permute.default(add_13, [0, 2, 3, 1]);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_14: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_6, torch.float32);  permute_6 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_14, [3], correction = 0, keepdim = True)
        getitem_6: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_7);  convert_element_type_14 = getitem_7 = None
        add_14: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
        rsqrt_3: "f32[128, 72, 72, 1][5184, 72, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_14: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_15: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, arg25_1);  mul_14 = arg25_1 = None
        add_15: "f32[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, arg26_1);  mul_15 = arg26_1 = None
        convert_element_type_15: "bf16[128, 72, 72, 80][414720, 5760, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_7: "bf16[128, 80, 72, 72][414720, 1, 5760, 80]cuda:0" = torch.ops.aten.permute.default(convert_element_type_15, [0, 3, 1, 2]);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        convolution_7: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.convolution.default(permute_7, arg27_1, arg28_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_7 = arg27_1 = arg28_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_8: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.convolution.default(convolution_7, arg29_1, arg30_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 160);  arg29_1 = arg30_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_8: "bf16[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_8, [0, 2, 3, 1]);  convolution_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_16: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_8, torch.float32);  permute_8 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_16, [3], correction = 0, keepdim = True)
        getitem_8: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_4: "bf16[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg35_1, [1, -1, 1, 1]);  arg35_1 = None
        view_5: "bf16[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg36_1, [1, -1, 1, 1]);  arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_4: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_9);  convert_element_type_16 = getitem_9 = None
        add_16: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
        rsqrt_4: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_16: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, arg31_1);  mul_16 = arg31_1 = None
        add_17: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, arg32_1);  mul_17 = arg32_1 = None
        convert_element_type_17: "bf16[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_9: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.permute.default(convert_element_type_17, [0, 3, 1, 2]);  convert_element_type_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_9: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.convolution.default(permute_9, arg33_1, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_9 = arg33_1 = arg34_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_18: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        mul_18: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.5)
        mul_19: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.7071067811865476);  convert_element_type_18 = None
        erf_2: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_18: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, add_18);  mul_18 = add_18 = None
        convert_element_type_19: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_20: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_19, torch.float32)
        pow_5: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_20, 2);  convert_element_type_20 = None
        sum_3: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_5, [2, 3], True);  pow_5 = None
        pow_6: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_3, 0.5);  sum_3 = None
        convert_element_type_21: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_6, torch.bfloat16);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_2: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_21, [1], True)
        add_19: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        div_2: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_21, add_19);  convert_element_type_21 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_21: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, div_2);  div_2 = None
        addcmul_2: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.addcmul.default(view_4, view_5, mul_21);  view_4 = view_5 = mul_21 = None
        add_20: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_19, addcmul_2);  convert_element_type_19 = addcmul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_10: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.convolution.default(add_20, arg37_1, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_20 = arg37_1 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_21: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.add.Tensor(convolution_10, convolution_7);  convolution_10 = convolution_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_11: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.convolution.default(add_21, arg39_1, arg40_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 160);  arg39_1 = arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_10: "bf16[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_22: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_10, torch.float32);  permute_10 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_22, [3], correction = 0, keepdim = True)
        getitem_10: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_6: "bf16[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg45_1, [1, -1, 1, 1]);  arg45_1 = None
        view_7: "bf16[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg46_1, [1, -1, 1, 1]);  arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_5: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_22, getitem_11);  convert_element_type_22 = getitem_11 = None
        add_22: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
        rsqrt_5: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_22: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_23: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg41_1);  mul_22 = arg41_1 = None
        add_23: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg42_1);  mul_23 = arg42_1 = None
        convert_element_type_23: "bf16[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_11: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.permute.default(convert_element_type_23, [0, 3, 1, 2]);  convert_element_type_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_12: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.convolution.default(permute_11, arg43_1, arg44_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_11 = arg43_1 = arg44_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_24: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        mul_24: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5)
        mul_25: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.7071067811865476);  convert_element_type_24 = None
        erf_3: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_24: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_26: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_24);  mul_24 = add_24 = None
        convert_element_type_25: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_26: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_25, torch.float32)
        pow_7: "f32[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_26, 2);  convert_element_type_26 = None
        sum_4: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_7, [2, 3], True);  pow_7 = None
        pow_8: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
        convert_element_type_27: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_8, torch.bfloat16);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_3: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_27, [1], True)
        add_25: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        div_3: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_27, add_25);  convert_element_type_27 = add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_27: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, div_3);  div_3 = None
        addcmul_3: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.addcmul.default(view_6, view_7, mul_27);  view_6 = view_7 = mul_27 = None
        add_26: "bf16[128, 640, 36, 36][829440, 1, 23040, 640]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_25, addcmul_3);  convert_element_type_25 = addcmul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_13: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.convolution.default(add_26, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_26 = arg47_1 = arg48_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_27: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.add.Tensor(convolution_13, add_21);  convolution_13 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_12: "bf16[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.permute.default(add_27, [0, 2, 3, 1]);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_28: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_12, torch.float32);  permute_12 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_28, [3], correction = 0, keepdim = True)
        getitem_12: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_28, getitem_13);  convert_element_type_28 = getitem_13 = None
        add_28: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        rsqrt_6: "f32[128, 36, 36, 1][1296, 36, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_28: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_29: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, arg49_1);  mul_28 = arg49_1 = None
        add_29: "f32[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, arg50_1);  mul_29 = arg50_1 = None
        convert_element_type_29: "bf16[128, 36, 36, 160][207360, 5760, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_13: "bf16[128, 160, 36, 36][207360, 1, 5760, 160]cuda:0" = torch.ops.aten.permute.default(convert_element_type_29, [0, 3, 1, 2]);  convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        convolution_14: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(permute_13, arg51_1, arg52_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_13 = arg51_1 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_15: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(convolution_14, arg53_1, arg54_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg53_1 = arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_14: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_30: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_30, [3], correction = 0, keepdim = True)
        getitem_14: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_8: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg59_1, [1, -1, 1, 1]);  arg59_1 = None
        view_9: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg60_1, [1, -1, 1, 1]);  arg60_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_7: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_15);  convert_element_type_30 = getitem_15 = None
        add_30: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_7: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_30: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_31: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, arg55_1);  mul_30 = arg55_1 = None
        add_31: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, arg56_1);  mul_31 = arg56_1 = None
        convert_element_type_31: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_15: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.permute.default(convert_element_type_31, [0, 3, 1, 2]);  convert_element_type_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_16: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_15, arg57_1, arg58_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_15 = arg57_1 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_32: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        mul_32: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_32, 0.5)
        mul_33: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_32, 0.7071067811865476);  convert_element_type_32 = None
        erf_4: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_32: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, add_32);  mul_32 = add_32 = None
        convert_element_type_33: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_34: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_33, torch.float32)
        pow_9: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_34, 2);  convert_element_type_34 = None
        sum_5: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_9, [2, 3], True);  pow_9 = None
        pow_10: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_5, 0.5);  sum_5 = None
        convert_element_type_35: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_10, torch.bfloat16);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_4: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_35, [1], True)
        add_33: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        div_4: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_35, add_33);  convert_element_type_35 = add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_35: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_33, div_4);  div_4 = None
        addcmul_4: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_8, view_9, mul_35);  view_8 = view_9 = mul_35 = None
        add_34: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_33, addcmul_4);  convert_element_type_33 = addcmul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_17: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_34, arg61_1, arg62_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_34 = arg61_1 = arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_35: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_17, convolution_14);  convolution_17 = convolution_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_18: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_35, arg63_1, arg64_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg63_1 = arg64_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_16: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1]);  convolution_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_36: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_16, torch.float32);  permute_16 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_36, [3], correction = 0, keepdim = True)
        getitem_16: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_10: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg69_1, [1, -1, 1, 1]);  arg69_1 = None
        view_11: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg70_1, [1, -1, 1, 1]);  arg70_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_8: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_36, getitem_17);  convert_element_type_36 = getitem_17 = None
        add_36: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_8: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_36: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_37: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg65_1);  mul_36 = arg65_1 = None
        add_37: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg66_1);  mul_37 = arg66_1 = None
        convert_element_type_37: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_17: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.permute.default(convert_element_type_37, [0, 3, 1, 2]);  convert_element_type_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_19: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_17, arg67_1, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_17 = arg67_1 = arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_38: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        mul_38: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5)
        mul_39: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.7071067811865476);  convert_element_type_38 = None
        erf_5: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_38: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_40: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_38);  mul_38 = add_38 = None
        convert_element_type_39: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_40: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_39, torch.float32)
        pow_11: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_40, 2);  convert_element_type_40 = None
        sum_6: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_11, [2, 3], True);  pow_11 = None
        pow_12: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_6, 0.5);  sum_6 = None
        convert_element_type_41: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_12, torch.bfloat16);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_5: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_41, [1], True)
        add_39: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        div_5: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_41, add_39);  convert_element_type_41 = add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_41: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, div_5);  div_5 = None
        addcmul_5: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_10, view_11, mul_41);  view_10 = view_11 = mul_41 = None
        add_40: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_39, addcmul_5);  convert_element_type_39 = addcmul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_20: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_40, arg71_1, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_40 = arg71_1 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_41: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_20, add_35);  convolution_20 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_21: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_41, arg73_1, arg74_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg73_1 = arg74_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_18: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_42: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_18, torch.float32);  permute_18 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_42, [3], correction = 0, keepdim = True)
        getitem_18: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_12: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg79_1, [1, -1, 1, 1]);  arg79_1 = None
        view_13: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg80_1, [1, -1, 1, 1]);  arg80_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_9: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_42, getitem_19);  convert_element_type_42 = getitem_19 = None
        add_42: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
        rsqrt_9: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_42: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_43: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, arg75_1);  mul_42 = arg75_1 = None
        add_43: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, arg76_1);  mul_43 = arg76_1 = None
        convert_element_type_43: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_19: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.permute.default(convert_element_type_43, [0, 3, 1, 2]);  convert_element_type_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_22: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_19, arg77_1, arg78_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_19 = arg77_1 = arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_44: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        mul_44: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.5)
        mul_45: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.7071067811865476);  convert_element_type_44 = None
        erf_6: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_44: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_46: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, add_44);  mul_44 = add_44 = None
        convert_element_type_45: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_46: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_45, torch.float32)
        pow_13: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_46, 2);  convert_element_type_46 = None
        sum_7: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_13, [2, 3], True);  pow_13 = None
        pow_14: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_7, 0.5);  sum_7 = None
        convert_element_type_47: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_14, torch.bfloat16);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_6: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_47, [1], True)
        add_45: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        div_6: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_47, add_45);  convert_element_type_47 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_47: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, div_6);  div_6 = None
        addcmul_6: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_12, view_13, mul_47);  view_12 = view_13 = mul_47 = None
        add_46: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_45, addcmul_6);  convert_element_type_45 = addcmul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_23: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_46, arg81_1, arg82_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_46 = arg81_1 = arg82_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_47: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_23, add_41);  convolution_23 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_24: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_47, arg83_1, arg84_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg83_1 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_20: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_48: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_20, torch.float32);  permute_20 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_48, [3], correction = 0, keepdim = True)
        getitem_20: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_14: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg89_1, [1, -1, 1, 1]);  arg89_1 = None
        view_15: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg90_1, [1, -1, 1, 1]);  arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_10: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_21);  convert_element_type_48 = getitem_21 = None
        add_48: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        rsqrt_10: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_48: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_49: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, arg85_1);  mul_48 = arg85_1 = None
        add_49: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, arg86_1);  mul_49 = arg86_1 = None
        convert_element_type_49: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_21: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.permute.default(convert_element_type_49, [0, 3, 1, 2]);  convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_25: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_21, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_21 = arg87_1 = arg88_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_50: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        mul_50: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_50, 0.5)
        mul_51: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_50, 0.7071067811865476);  convert_element_type_50 = None
        erf_7: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_50: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_52: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_50);  mul_50 = add_50 = None
        convert_element_type_51: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_52: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_51, torch.float32)
        pow_15: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_52, 2);  convert_element_type_52 = None
        sum_8: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_15, [2, 3], True);  pow_15 = None
        pow_16: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_8, 0.5);  sum_8 = None
        convert_element_type_53: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_16, torch.bfloat16);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_7: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_53, [1], True)
        add_51: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        div_7: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_53, add_51);  convert_element_type_53 = add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_53: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, div_7);  div_7 = None
        addcmul_7: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_14, view_15, mul_53);  view_14 = view_15 = mul_53 = None
        add_52: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_51, addcmul_7);  convert_element_type_51 = addcmul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_26: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_52, arg91_1, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_52 = arg91_1 = arg92_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_53: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_26, add_47);  convolution_26 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_27: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_53, arg93_1, arg94_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg93_1 = arg94_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_22: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_54: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_22, torch.float32);  permute_22 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_54, [3], correction = 0, keepdim = True)
        getitem_22: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_16: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg99_1, [1, -1, 1, 1]);  arg99_1 = None
        view_17: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg100_1, [1, -1, 1, 1]);  arg100_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_11: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_54, getitem_23);  convert_element_type_54 = getitem_23 = None
        add_54: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_11: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_54: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_55: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, arg95_1);  mul_54 = arg95_1 = None
        add_55: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, arg96_1);  mul_55 = arg96_1 = None
        convert_element_type_55: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_23: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.permute.default(convert_element_type_55, [0, 3, 1, 2]);  convert_element_type_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_28: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_23, arg97_1, arg98_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_23 = arg97_1 = arg98_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_56: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        mul_56: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, 0.5)
        mul_57: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, 0.7071067811865476);  convert_element_type_56 = None
        erf_8: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_56: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_58: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, add_56);  mul_56 = add_56 = None
        convert_element_type_57: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_58: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_57, torch.float32)
        pow_17: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_58, 2);  convert_element_type_58 = None
        sum_9: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_17, [2, 3], True);  pow_17 = None
        pow_18: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_9, 0.5);  sum_9 = None
        convert_element_type_59: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_18, torch.bfloat16);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_8: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_59, [1], True)
        add_57: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        div_8: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_59, add_57);  convert_element_type_59 = add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_59: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_57, div_8);  div_8 = None
        addcmul_8: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_16, view_17, mul_59);  view_16 = view_17 = mul_59 = None
        add_58: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_57, addcmul_8);  convert_element_type_57 = addcmul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_29: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_58, arg101_1, arg102_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_58 = arg101_1 = arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_59: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_29, add_53);  convolution_29 = add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_30: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_59, arg103_1, arg104_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg103_1 = arg104_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_24: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1]);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_60: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_24, torch.float32);  permute_24 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_60, [3], correction = 0, keepdim = True)
        getitem_24: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_18: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg109_1, [1, -1, 1, 1]);  arg109_1 = None
        view_19: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg110_1, [1, -1, 1, 1]);  arg110_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_12: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_25);  convert_element_type_60 = getitem_25 = None
        add_60: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_12: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_60: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_61: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, arg105_1);  mul_60 = arg105_1 = None
        add_61: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, arg106_1);  mul_61 = arg106_1 = None
        convert_element_type_61: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_25: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.permute.default(convert_element_type_61, [0, 3, 1, 2]);  convert_element_type_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_31: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_25, arg107_1, arg108_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_25 = arg107_1 = arg108_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_62: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        mul_62: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.5)
        mul_63: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.7071067811865476);  convert_element_type_62 = None
        erf_9: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_62: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_64: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, add_62);  mul_62 = add_62 = None
        convert_element_type_63: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_64: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_63, torch.float32)
        pow_19: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_64, 2);  convert_element_type_64 = None
        sum_10: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_19, [2, 3], True);  pow_19 = None
        pow_20: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None
        convert_element_type_65: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_20, torch.bfloat16);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_9: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_65, [1], True)
        add_63: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        div_9: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_65, add_63);  convert_element_type_65 = add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_65: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_63, div_9);  div_9 = None
        addcmul_9: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_18, view_19, mul_65);  view_18 = view_19 = mul_65 = None
        add_64: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_63, addcmul_9);  convert_element_type_63 = addcmul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_32: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_64, arg111_1, arg112_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_64 = arg111_1 = arg112_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_65: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_32, add_59);  convolution_32 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_33: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_65, arg113_1, arg114_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg113_1 = arg114_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_26: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_66: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_26, torch.float32);  permute_26 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_66, [3], correction = 0, keepdim = True)
        getitem_26: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_20: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg119_1, [1, -1, 1, 1]);  arg119_1 = None
        view_21: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg120_1, [1, -1, 1, 1]);  arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_13: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_66, getitem_27);  convert_element_type_66 = getitem_27 = None
        add_66: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        rsqrt_13: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_66: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_67: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg115_1);  mul_66 = arg115_1 = None
        add_67: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg116_1);  mul_67 = arg116_1 = None
        convert_element_type_67: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_27: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.permute.default(convert_element_type_67, [0, 3, 1, 2]);  convert_element_type_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_34: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_27, arg117_1, arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_27 = arg117_1 = arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_68: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        mul_68: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.5)
        mul_69: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.7071067811865476);  convert_element_type_68 = None
        erf_10: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_68: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_70: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_68);  mul_68 = add_68 = None
        convert_element_type_69: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_70: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_69, torch.float32)
        pow_21: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_70, 2);  convert_element_type_70 = None
        sum_11: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_21, [2, 3], True);  pow_21 = None
        pow_22: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_11, 0.5);  sum_11 = None
        convert_element_type_71: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_22, torch.bfloat16);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_10: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_71, [1], True)
        add_69: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        div_10: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_71, add_69);  convert_element_type_71 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_71: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_69, div_10);  div_10 = None
        addcmul_10: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_20, view_21, mul_71);  view_20 = view_21 = mul_71 = None
        add_70: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_69, addcmul_10);  convert_element_type_69 = addcmul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_35: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_70, arg121_1, arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_70 = arg121_1 = arg122_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_71: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_35, add_65);  convolution_35 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_36: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_71, arg123_1, arg124_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg123_1 = arg124_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_28: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_36, [0, 2, 3, 1]);  convolution_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_72: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_28, torch.float32);  permute_28 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_72, [3], correction = 0, keepdim = True)
        getitem_28: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_22: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg129_1, [1, -1, 1, 1]);  arg129_1 = None
        view_23: "bf16[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg130_1, [1, -1, 1, 1]);  arg130_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_14: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_29);  convert_element_type_72 = getitem_29 = None
        add_72: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_14: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_72: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_73: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, arg125_1);  mul_72 = arg125_1 = None
        add_73: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, arg126_1);  mul_73 = arg126_1 = None
        convert_element_type_73: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_29: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.permute.default(convert_element_type_73, [0, 3, 1, 2]);  convert_element_type_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_37: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_29, arg127_1, arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_29 = arg127_1 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_74: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        mul_74: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.5)
        mul_75: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.7071067811865476);  convert_element_type_74 = None
        erf_11: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_74: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_76: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, add_74);  mul_74 = add_74 = None
        convert_element_type_75: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_76: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_75, torch.float32)
        pow_23: "f32[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_76, 2);  convert_element_type_76 = None
        sum_12: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_23, [2, 3], True);  pow_23 = None
        pow_24: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_12, 0.5);  sum_12 = None
        convert_element_type_77: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_24, torch.bfloat16);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_11: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_77, [1], True)
        add_75: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        div_11: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_77, add_75);  convert_element_type_77 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_77: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, div_11);  div_11 = None
        addcmul_11: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_22, view_23, mul_77);  view_22 = view_23 = mul_77 = None
        add_76: "bf16[128, 1280, 18, 18][414720, 1, 23040, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_75, addcmul_11);  convert_element_type_75 = addcmul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_38: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.convolution.default(add_76, arg131_1, arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_76 = arg131_1 = arg132_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_77: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_38, add_71);  convolution_38 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_30: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.permute.default(add_77, [0, 2, 3, 1]);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_78: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_30, torch.float32);  permute_30 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_78, [3], correction = 0, keepdim = True)
        getitem_30: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_78, getitem_31);  convert_element_type_78 = getitem_31 = None
        add_78: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_15: "f32[128, 18, 18, 1][324, 18, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_78: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_79: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, arg133_1);  mul_78 = arg133_1 = None
        add_79: "f32[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, arg134_1);  mul_79 = arg134_1 = None
        convert_element_type_79: "bf16[128, 18, 18, 320][103680, 5760, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_31: "bf16[128, 320, 18, 18][103680, 1, 5760, 320]cuda:0" = torch.ops.aten.permute.default(convert_element_type_79, [0, 3, 1, 2]);  convert_element_type_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        convolution_39: "bf16[128, 640, 9, 9][51840, 1, 5760, 640]cuda:0" = torch.ops.aten.convolution.default(permute_31, arg135_1, arg136_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_31 = arg135_1 = arg136_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_40: "bf16[128, 640, 9, 9][51840, 1, 5760, 640]cuda:0" = torch.ops.aten.convolution.default(convolution_39, arg137_1, arg138_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 640);  arg137_1 = arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_32: "bf16[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_40, [0, 2, 3, 1]);  convolution_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_80: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_32, torch.float32);  permute_32 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_80, [3], correction = 0, keepdim = True)
        getitem_32: "f32[128, 9, 9, 1][81, 9, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[128, 9, 9, 1][81, 9, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_24: "bf16[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg143_1, [1, -1, 1, 1]);  arg143_1 = None
        view_25: "bf16[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg144_1, [1, -1, 1, 1]);  arg144_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_16: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_33);  convert_element_type_80 = getitem_33 = None
        add_80: "f32[128, 9, 9, 1][81, 9, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_16: "f32[128, 9, 9, 1][81, 9, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_80: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_81: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, arg139_1);  mul_80 = arg139_1 = None
        add_81: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, arg140_1);  mul_81 = arg140_1 = None
        convert_element_type_81: "bf16[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_33: "bf16[128, 640, 9, 9][51840, 1, 5760, 640]cuda:0" = torch.ops.aten.permute.default(convert_element_type_81, [0, 3, 1, 2]);  convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_41: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.convolution.default(permute_33, arg141_1, arg142_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_33 = arg141_1 = arg142_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_82: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        mul_82: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_82, 0.5)
        mul_83: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_82, 0.7071067811865476);  convert_element_type_82 = None
        erf_12: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_82: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_84: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, add_82);  mul_82 = add_82 = None
        convert_element_type_83: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_84: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_83, torch.float32)
        pow_25: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_84, 2);  convert_element_type_84 = None
        sum_13: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_25, [2, 3], True);  pow_25 = None
        pow_26: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_13, 0.5);  sum_13 = None
        convert_element_type_85: "bf16[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_26, torch.bfloat16);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_12: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_85, [1], True)
        add_83: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        div_12: "bf16[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_85, add_83);  convert_element_type_85 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_85: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_83, div_12);  div_12 = None
        addcmul_12: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.addcmul.default(view_24, view_25, mul_85);  view_24 = view_25 = mul_85 = None
        add_84: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_83, addcmul_12);  convert_element_type_83 = addcmul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_42: "bf16[128, 640, 9, 9][51840, 1, 5760, 640]cuda:0" = torch.ops.aten.convolution.default(add_84, arg145_1, arg146_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_84 = arg145_1 = arg146_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_85: "bf16[128, 640, 9, 9][51840, 1, 5760, 640]cuda:0" = torch.ops.aten.add.Tensor(convolution_42, convolution_39);  convolution_42 = convolution_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_43: "bf16[128, 640, 9, 9][51840, 1, 5760, 640]cuda:0" = torch.ops.aten.convolution.default(add_85, arg147_1, arg148_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 640);  arg147_1 = arg148_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_34: "bf16[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1]);  convolution_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_86: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_34, torch.float32);  permute_34 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_86, [3], correction = 0, keepdim = True)
        getitem_34: "f32[128, 9, 9, 1][81, 9, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[128, 9, 9, 1][81, 9, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_26: "bf16[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg153_1, [1, -1, 1, 1]);  arg153_1 = None
        view_27: "bf16[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg154_1, [1, -1, 1, 1]);  arg154_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_17: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_35);  convert_element_type_86 = getitem_35 = None
        add_86: "f32[128, 9, 9, 1][81, 9, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
        rsqrt_17: "f32[128, 9, 9, 1][81, 9, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_86: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_87: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, arg149_1);  mul_86 = arg149_1 = None
        add_87: "f32[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, arg150_1);  mul_87 = arg150_1 = None
        convert_element_type_87: "bf16[128, 9, 9, 640][51840, 5760, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_35: "bf16[128, 640, 9, 9][51840, 1, 5760, 640]cuda:0" = torch.ops.aten.permute.default(convert_element_type_87, [0, 3, 1, 2]);  convert_element_type_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_44: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.convolution.default(permute_35, arg151_1, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_35 = arg151_1 = arg152_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_88: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        mul_88: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.5)
        mul_89: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.7071067811865476);  convert_element_type_88 = None
        erf_13: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_88: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_90: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, add_88);  mul_88 = add_88 = None
        convert_element_type_89: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_90: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_89, torch.float32)
        pow_27: "f32[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_90, 2);  convert_element_type_90 = None
        sum_14: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_27, [2, 3], True);  pow_27 = None
        pow_28: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_14, 0.5);  sum_14 = None
        convert_element_type_91: "bf16[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_28, torch.bfloat16);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_13: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_91, [1], True)
        add_89: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        div_13: "bf16[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_91, add_89);  convert_element_type_91 = add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_91: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_89, div_13);  div_13 = None
        addcmul_13: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.addcmul.default(view_26, view_27, mul_91);  view_26 = view_27 = mul_91 = None
        add_90: "bf16[128, 2560, 9, 9][207360, 1, 23040, 2560]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_89, addcmul_13);  convert_element_type_89 = addcmul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_45: "bf16[128, 640, 9, 9][51840, 1, 5760, 640]cuda:0" = torch.ops.aten.convolution.default(add_90, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_90 = arg155_1 = arg156_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_91: "bf16[128, 640, 9, 9][51840, 1, 5760, 640]cuda:0" = torch.ops.aten.add.Tensor(convolution_45, add_85);  convolution_45 = add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_14: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_91, [-1, -2], True);  add_91 = None
        as_strided: "bf16[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.as_strided.default(mean_14, [128, 640, 1, 1], [640, 1, 640, 640]);  mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_36: "bf16[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.permute.default(as_strided, [0, 2, 3, 1]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_92: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_36, torch.float32);  permute_36 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_92, [3], correction = 0, keepdim = True)
        getitem_36: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_92, getitem_37);  convert_element_type_92 = getitem_37 = None
        add_92: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_18: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_92: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_93: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, arg157_1);  mul_92 = arg157_1 = None
        add_93: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, arg158_1);  mul_93 = arg158_1 = None
        convert_element_type_93: "bf16[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_37: "bf16[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.permute.default(convert_element_type_93, [0, 3, 1, 2]);  convert_element_type_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:214 in forward, code: x = self.flatten(x)
        view_28: "bf16[128, 640][640, 1]cuda:0" = torch.ops.aten.reshape.default(permute_37, [128, 640]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:219 in forward, code: x = self.fc(x)
        permute_38: "bf16[640, 1000][1, 640]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_28, permute_38);  arg160_1 = view_28 = permute_38 = None
        return (addmm,)

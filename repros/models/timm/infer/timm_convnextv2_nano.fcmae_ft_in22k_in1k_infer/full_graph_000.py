import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[80, 3, 4, 4][48, 1, 12, 3]cuda:0", arg1_1: "f32[80][1]cuda:0", arg2_1: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", arg3_1: "f32[80][1]cuda:0", arg4_1: "f32[80][1]cuda:0", arg5_1: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0", arg6_1: "f32[80][1]cuda:0", arg7_1: "f32[80][1]cuda:0", arg8_1: "f32[80][1]cuda:0", arg9_1: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0", arg10_1: "f32[320][1]cuda:0", arg11_1: "f32[320][1]cuda:0", arg12_1: "f32[320][1]cuda:0", arg13_1: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0", arg14_1: "f32[80][1]cuda:0", arg15_1: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0", arg16_1: "f32[80][1]cuda:0", arg17_1: "f32[80][1]cuda:0", arg18_1: "f32[80][1]cuda:0", arg19_1: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0", arg20_1: "f32[320][1]cuda:0", arg21_1: "f32[320][1]cuda:0", arg22_1: "f32[320][1]cuda:0", arg23_1: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0", arg24_1: "f32[80][1]cuda:0", arg25_1: "f32[80][1]cuda:0", arg26_1: "f32[80][1]cuda:0", arg27_1: "f32[160, 80, 2, 2][320, 1, 160, 80]cuda:0", arg28_1: "f32[160][1]cuda:0", arg29_1: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0", arg30_1: "f32[160][1]cuda:0", arg31_1: "f32[160][1]cuda:0", arg32_1: "f32[160][1]cuda:0", arg33_1: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0", arg34_1: "f32[640][1]cuda:0", arg35_1: "f32[640][1]cuda:0", arg36_1: "f32[640][1]cuda:0", arg37_1: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0", arg38_1: "f32[160][1]cuda:0", arg39_1: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0", arg40_1: "f32[160][1]cuda:0", arg41_1: "f32[160][1]cuda:0", arg42_1: "f32[160][1]cuda:0", arg43_1: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0", arg44_1: "f32[640][1]cuda:0", arg45_1: "f32[640][1]cuda:0", arg46_1: "f32[640][1]cuda:0", arg47_1: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0", arg48_1: "f32[160][1]cuda:0", arg49_1: "f32[160][1]cuda:0", arg50_1: "f32[160][1]cuda:0", arg51_1: "f32[320, 160, 2, 2][640, 1, 320, 160]cuda:0", arg52_1: "f32[320][1]cuda:0", arg53_1: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg54_1: "f32[320][1]cuda:0", arg55_1: "f32[320][1]cuda:0", arg56_1: "f32[320][1]cuda:0", arg57_1: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg58_1: "f32[1280][1]cuda:0", arg59_1: "f32[1280][1]cuda:0", arg60_1: "f32[1280][1]cuda:0", arg61_1: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg62_1: "f32[320][1]cuda:0", arg63_1: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg64_1: "f32[320][1]cuda:0", arg65_1: "f32[320][1]cuda:0", arg66_1: "f32[320][1]cuda:0", arg67_1: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg68_1: "f32[1280][1]cuda:0", arg69_1: "f32[1280][1]cuda:0", arg70_1: "f32[1280][1]cuda:0", arg71_1: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg72_1: "f32[320][1]cuda:0", arg73_1: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg74_1: "f32[320][1]cuda:0", arg75_1: "f32[320][1]cuda:0", arg76_1: "f32[320][1]cuda:0", arg77_1: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg78_1: "f32[1280][1]cuda:0", arg79_1: "f32[1280][1]cuda:0", arg80_1: "f32[1280][1]cuda:0", arg81_1: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg82_1: "f32[320][1]cuda:0", arg83_1: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg84_1: "f32[320][1]cuda:0", arg85_1: "f32[320][1]cuda:0", arg86_1: "f32[320][1]cuda:0", arg87_1: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg88_1: "f32[1280][1]cuda:0", arg89_1: "f32[1280][1]cuda:0", arg90_1: "f32[1280][1]cuda:0", arg91_1: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg92_1: "f32[320][1]cuda:0", arg93_1: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg94_1: "f32[320][1]cuda:0", arg95_1: "f32[320][1]cuda:0", arg96_1: "f32[320][1]cuda:0", arg97_1: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg98_1: "f32[1280][1]cuda:0", arg99_1: "f32[1280][1]cuda:0", arg100_1: "f32[1280][1]cuda:0", arg101_1: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg102_1: "f32[320][1]cuda:0", arg103_1: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg104_1: "f32[320][1]cuda:0", arg105_1: "f32[320][1]cuda:0", arg106_1: "f32[320][1]cuda:0", arg107_1: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg108_1: "f32[1280][1]cuda:0", arg109_1: "f32[1280][1]cuda:0", arg110_1: "f32[1280][1]cuda:0", arg111_1: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg112_1: "f32[320][1]cuda:0", arg113_1: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg114_1: "f32[320][1]cuda:0", arg115_1: "f32[320][1]cuda:0", arg116_1: "f32[320][1]cuda:0", arg117_1: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg118_1: "f32[1280][1]cuda:0", arg119_1: "f32[1280][1]cuda:0", arg120_1: "f32[1280][1]cuda:0", arg121_1: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg122_1: "f32[320][1]cuda:0", arg123_1: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", arg124_1: "f32[320][1]cuda:0", arg125_1: "f32[320][1]cuda:0", arg126_1: "f32[320][1]cuda:0", arg127_1: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg128_1: "f32[1280][1]cuda:0", arg129_1: "f32[1280][1]cuda:0", arg130_1: "f32[1280][1]cuda:0", arg131_1: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", arg132_1: "f32[320][1]cuda:0", arg133_1: "f32[320][1]cuda:0", arg134_1: "f32[320][1]cuda:0", arg135_1: "f32[640, 320, 2, 2][1280, 1, 640, 320]cuda:0", arg136_1: "f32[640][1]cuda:0", arg137_1: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0", arg138_1: "f32[640][1]cuda:0", arg139_1: "f32[640][1]cuda:0", arg140_1: "f32[640][1]cuda:0", arg141_1: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", arg142_1: "f32[2560][1]cuda:0", arg143_1: "f32[2560][1]cuda:0", arg144_1: "f32[2560][1]cuda:0", arg145_1: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", arg146_1: "f32[640][1]cuda:0", arg147_1: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0", arg148_1: "f32[640][1]cuda:0", arg149_1: "f32[640][1]cuda:0", arg150_1: "f32[640][1]cuda:0", arg151_1: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", arg152_1: "f32[2560][1]cuda:0", arg153_1: "f32[2560][1]cuda:0", arg154_1: "f32[2560][1]cuda:0", arg155_1: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", arg156_1: "f32[640][1]cuda:0", arg157_1: "f32[640][1]cuda:0", arg158_1: "f32[640][1]cuda:0", arg159_1: "f32[1000, 640][640, 1]cuda:0", arg160_1: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:608 in forward_features, code: x = self.stem(x)
        convolution: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean = torch.ops.aten.var_mean.correction(permute, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute, getitem_1);  permute = getitem_1 = None
        add: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_1: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.permute.default(add_1, [0, 3, 1, 2]);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_1: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(permute_1, arg5_1, arg6_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 80);  arg5_1 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_2: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_1 = torch.ops.aten.var_mean.correction(permute_2, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg11_1, [1, -1, 1, 1]);  arg11_1 = None
        view_1: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg12_1, [1, -1, 1, 1]);  arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_1: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_2, getitem_3);  permute_2 = getitem_3 = None
        add_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
        add_3: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, arg8_1);  mul_3 = arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_3: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.permute.default(add_3, [0, 3, 1, 2]);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_2: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.convolution.default(permute_3, arg9_1, arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_3 = arg9_1 = arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_4: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_2, 0.5)
        mul_5: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476);  convolution_2 = None
        erf: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_1: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_6, 2)
        sum_1: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_1, [2, 3], True);  pow_1 = None
        pow_2: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_2, [1], True)
        add_5: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        div: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_2, add_5);  pow_2 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_7: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, div);  div = None
        addcmul: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.addcmul.default(view, view_1, mul_7);  view = view_1 = mul_7 = None
        add_6: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_6, addcmul);  mul_6 = addcmul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_3: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(add_6, arg13_1, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_6 = arg13_1 = arg14_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_7: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.add.Tensor(convolution_3, permute_1);  convolution_3 = permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_4: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(add_7, arg15_1, arg16_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 80);  arg15_1 = arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_4: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_2 = torch.ops.aten.var_mean.correction(permute_4, [3], correction = 0, keepdim = True)
        getitem_4: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_2: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg21_1, [1, -1, 1, 1]);  arg21_1 = None
        view_3: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg22_1, [1, -1, 1, 1]);  arg22_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_2: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_4, getitem_5);  permute_4 = getitem_5 = None
        add_8: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
        rsqrt_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_8: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, arg17_1);  mul_8 = arg17_1 = None
        add_9: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, arg18_1);  mul_9 = arg18_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_5: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.permute.default(add_9, [0, 3, 1, 2]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_5: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.convolution.default(permute_5, arg19_1, arg20_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_5 = arg19_1 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_10: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_5, 0.5)
        mul_11: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476);  convolution_5 = None
        erf_1: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_10: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, add_10);  mul_10 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_3: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_12, 2)
        sum_2: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_3, [2, 3], True);  pow_3 = None
        pow_4: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_1: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_4, [1], True)
        add_11: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        div_1: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_4, add_11);  pow_4 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_13: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, div_1);  div_1 = None
        addcmul_1: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.addcmul.default(view_2, view_3, mul_13);  view_2 = view_3 = mul_13 = None
        add_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_12, addcmul_1);  mul_12 = addcmul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_6: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(add_12, arg23_1, arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_12 = arg23_1 = arg24_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_13: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.add.Tensor(convolution_6, add_7);  convolution_6 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_6: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(add_13, [0, 2, 3, 1]);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_3 = torch.ops.aten.var_mean.correction(permute_6, [3], correction = 0, keepdim = True)
        getitem_6: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_6, getitem_7);  permute_6 = getitem_7 = None
        add_14: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
        rsqrt_3: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_14: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_15: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, arg25_1);  mul_14 = arg25_1 = None
        add_15: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, arg26_1);  mul_15 = arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_7: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.permute.default(add_15, [0, 3, 1, 2]);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        convolution_7: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(permute_7, arg27_1, arg28_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_7 = arg27_1 = arg28_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_8: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(convolution_7, arg29_1, arg30_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 160);  arg29_1 = arg30_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_8: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_8, [0, 2, 3, 1]);  convolution_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_4 = torch.ops.aten.var_mean.correction(permute_8, [3], correction = 0, keepdim = True)
        getitem_8: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_4: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg35_1, [1, -1, 1, 1]);  arg35_1 = None
        view_5: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg36_1, [1, -1, 1, 1]);  arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_4: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_8, getitem_9);  permute_8 = getitem_9 = None
        add_16: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
        rsqrt_4: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_16: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, arg31_1);  mul_16 = arg31_1 = None
        add_17: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, arg32_1);  mul_17 = arg32_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_9: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.permute.default(add_17, [0, 3, 1, 2]);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_9: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.convolution.default(permute_9, arg33_1, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_9 = arg33_1 = arg34_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_18: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_9, 0.5)
        mul_19: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476);  convolution_9 = None
        erf_2: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_18: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, add_18);  mul_18 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_5: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_20, 2)
        sum_3: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_5, [2, 3], True);  pow_5 = None
        pow_6: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_3, 0.5);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_2: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_6, [1], True)
        add_19: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        div_2: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_6, add_19);  pow_6 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_21: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, div_2);  div_2 = None
        addcmul_2: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.addcmul.default(view_4, view_5, mul_21);  view_4 = view_5 = mul_21 = None
        add_20: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_20, addcmul_2);  mul_20 = addcmul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_10: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(add_20, arg37_1, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_20 = arg37_1 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_21: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.add.Tensor(convolution_10, convolution_7);  convolution_10 = convolution_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_11: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(add_21, arg39_1, arg40_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 160);  arg39_1 = arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_10: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_5 = torch.ops.aten.var_mean.correction(permute_10, [3], correction = 0, keepdim = True)
        getitem_10: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_6: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg45_1, [1, -1, 1, 1]);  arg45_1 = None
        view_7: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg46_1, [1, -1, 1, 1]);  arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_5: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_10, getitem_11);  permute_10 = getitem_11 = None
        add_22: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
        rsqrt_5: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_22: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_23: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg41_1);  mul_22 = arg41_1 = None
        add_23: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg42_1);  mul_23 = arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_11: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.permute.default(add_23, [0, 3, 1, 2]);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_12: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.convolution.default(permute_11, arg43_1, arg44_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_11 = arg43_1 = arg44_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_24: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_12, 0.5)
        mul_25: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476);  convolution_12 = None
        erf_3: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_24: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_26: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_24);  mul_24 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_7: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_26, 2)
        sum_4: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_7, [2, 3], True);  pow_7 = None
        pow_8: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_3: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_8, [1], True)
        add_25: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        div_3: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_8, add_25);  pow_8 = add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_27: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, div_3);  div_3 = None
        addcmul_3: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.addcmul.default(view_6, view_7, mul_27);  view_6 = view_7 = mul_27 = None
        add_26: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_26, addcmul_3);  mul_26 = addcmul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_13: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(add_26, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_26 = arg47_1 = arg48_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_27: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.add.Tensor(convolution_13, add_21);  convolution_13 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_12: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(add_27, [0, 2, 3, 1]);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_6 = torch.ops.aten.var_mean.correction(permute_12, [3], correction = 0, keepdim = True)
        getitem_12: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_12, getitem_13);  permute_12 = getitem_13 = None
        add_28: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        rsqrt_6: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_28: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_29: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, arg49_1);  mul_28 = arg49_1 = None
        add_29: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, arg50_1);  mul_29 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_13: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.permute.default(add_29, [0, 3, 1, 2]);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        convolution_14: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(permute_13, arg51_1, arg52_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_13 = arg51_1 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_15: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convolution_14, arg53_1, arg54_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg53_1 = arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_14: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_7 = torch.ops.aten.var_mean.correction(permute_14, [3], correction = 0, keepdim = True)
        getitem_14: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_8: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg59_1, [1, -1, 1, 1]);  arg59_1 = None
        view_9: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg60_1, [1, -1, 1, 1]);  arg60_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_7: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_14, getitem_15);  permute_14 = getitem_15 = None
        add_30: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_7: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_30: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_31: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, arg55_1);  mul_30 = arg55_1 = None
        add_31: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, arg56_1);  mul_31 = arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_15: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_31, [0, 3, 1, 2]);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_16: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_15, arg57_1, arg58_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_15 = arg57_1 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_32: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_16, 0.5)
        mul_33: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_16, 0.7071067811865476);  convolution_16 = None
        erf_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_32: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, add_32);  mul_32 = add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_34, 2)
        sum_5: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_9, [2, 3], True);  pow_9 = None
        pow_10: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_5, 0.5);  sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_4: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_10, [1], True)
        add_33: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        div_4: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_10, add_33);  pow_10 = add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_35: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, div_4);  div_4 = None
        addcmul_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_8, view_9, mul_35);  view_8 = view_9 = mul_35 = None
        add_34: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_34, addcmul_4);  mul_34 = addcmul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_17: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_34, arg61_1, arg62_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_34 = arg61_1 = arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_35: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_17, convolution_14);  convolution_17 = convolution_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_18: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_35, arg63_1, arg64_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg63_1 = arg64_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_16: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1]);  convolution_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_8 = torch.ops.aten.var_mean.correction(permute_16, [3], correction = 0, keepdim = True)
        getitem_16: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_10: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg69_1, [1, -1, 1, 1]);  arg69_1 = None
        view_11: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg70_1, [1, -1, 1, 1]);  arg70_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_8: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_16, getitem_17);  permute_16 = getitem_17 = None
        add_36: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_8: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_36: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_37: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg65_1);  mul_36 = arg65_1 = None
        add_37: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg66_1);  mul_37 = arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_17: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_37, [0, 3, 1, 2]);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_19: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_17, arg67_1, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_17 = arg67_1 = arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_38: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_19, 0.5)
        mul_39: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_19, 0.7071067811865476);  convolution_19 = None
        erf_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_38: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_40: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_38);  mul_38 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_11: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_40, 2)
        sum_6: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_11, [2, 3], True);  pow_11 = None
        pow_12: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_6, 0.5);  sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_5: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_12, [1], True)
        add_39: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        div_5: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_12, add_39);  pow_12 = add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_41: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, div_5);  div_5 = None
        addcmul_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_10, view_11, mul_41);  view_10 = view_11 = mul_41 = None
        add_40: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_40, addcmul_5);  mul_40 = addcmul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_20: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_40, arg71_1, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_40 = arg71_1 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_41: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_20, add_35);  convolution_20 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_21: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_41, arg73_1, arg74_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg73_1 = arg74_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_18: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_9 = torch.ops.aten.var_mean.correction(permute_18, [3], correction = 0, keepdim = True)
        getitem_18: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_12: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg79_1, [1, -1, 1, 1]);  arg79_1 = None
        view_13: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg80_1, [1, -1, 1, 1]);  arg80_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_9: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_18, getitem_19);  permute_18 = getitem_19 = None
        add_42: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
        rsqrt_9: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_42: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_43: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, arg75_1);  mul_42 = arg75_1 = None
        add_43: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, arg76_1);  mul_43 = arg76_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_19: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_43, [0, 3, 1, 2]);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_22: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_19, arg77_1, arg78_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_19 = arg77_1 = arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_44: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_22, 0.5)
        mul_45: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_22, 0.7071067811865476);  convolution_22 = None
        erf_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_44: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_46: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, add_44);  mul_44 = add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_13: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_46, 2)
        sum_7: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_13, [2, 3], True);  pow_13 = None
        pow_14: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_7, 0.5);  sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_6: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_14, [1], True)
        add_45: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        div_6: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_14, add_45);  pow_14 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_47: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, div_6);  div_6 = None
        addcmul_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_12, view_13, mul_47);  view_12 = view_13 = mul_47 = None
        add_46: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_46, addcmul_6);  mul_46 = addcmul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_23: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_46, arg81_1, arg82_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_46 = arg81_1 = arg82_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_47: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_23, add_41);  convolution_23 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_24: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_47, arg83_1, arg84_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg83_1 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_20: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_10 = torch.ops.aten.var_mean.correction(permute_20, [3], correction = 0, keepdim = True)
        getitem_20: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_14: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg89_1, [1, -1, 1, 1]);  arg89_1 = None
        view_15: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg90_1, [1, -1, 1, 1]);  arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_10: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_20, getitem_21);  permute_20 = getitem_21 = None
        add_48: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        rsqrt_10: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_48: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_49: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, arg85_1);  mul_48 = arg85_1 = None
        add_49: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, arg86_1);  mul_49 = arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_21: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_49, [0, 3, 1, 2]);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_25: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_21, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_21 = arg87_1 = arg88_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_50: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_25, 0.5)
        mul_51: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_25, 0.7071067811865476);  convolution_25 = None
        erf_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_50: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_52: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_50);  mul_50 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_15: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_52, 2)
        sum_8: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_15, [2, 3], True);  pow_15 = None
        pow_16: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_8, 0.5);  sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_7: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_16, [1], True)
        add_51: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        div_7: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_16, add_51);  pow_16 = add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_53: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, div_7);  div_7 = None
        addcmul_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_14, view_15, mul_53);  view_14 = view_15 = mul_53 = None
        add_52: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_52, addcmul_7);  mul_52 = addcmul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_26: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_52, arg91_1, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_52 = arg91_1 = arg92_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_53: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_26, add_47);  convolution_26 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_27: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_53, arg93_1, arg94_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg93_1 = arg94_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_22: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_11 = torch.ops.aten.var_mean.correction(permute_22, [3], correction = 0, keepdim = True)
        getitem_22: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_16: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg99_1, [1, -1, 1, 1]);  arg99_1 = None
        view_17: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg100_1, [1, -1, 1, 1]);  arg100_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_11: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_22, getitem_23);  permute_22 = getitem_23 = None
        add_54: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_11: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_54: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_55: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, arg95_1);  mul_54 = arg95_1 = None
        add_55: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, arg96_1);  mul_55 = arg96_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_23: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_55, [0, 3, 1, 2]);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_28: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_23, arg97_1, arg98_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_23 = arg97_1 = arg98_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_56: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_28, 0.5)
        mul_57: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_28, 0.7071067811865476);  convolution_28 = None
        erf_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_56: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_58: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, add_56);  mul_56 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_17: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_58, 2)
        sum_9: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_17, [2, 3], True);  pow_17 = None
        pow_18: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_9, 0.5);  sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_8: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_18, [1], True)
        add_57: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        div_8: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_18, add_57);  pow_18 = add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_59: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, div_8);  div_8 = None
        addcmul_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_16, view_17, mul_59);  view_16 = view_17 = mul_59 = None
        add_58: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_58, addcmul_8);  mul_58 = addcmul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_29: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_58, arg101_1, arg102_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_58 = arg101_1 = arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_59: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_29, add_53);  convolution_29 = add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_30: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_59, arg103_1, arg104_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg103_1 = arg104_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_24: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1]);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_12 = torch.ops.aten.var_mean.correction(permute_24, [3], correction = 0, keepdim = True)
        getitem_24: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_18: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg109_1, [1, -1, 1, 1]);  arg109_1 = None
        view_19: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg110_1, [1, -1, 1, 1]);  arg110_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_12: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_24, getitem_25);  permute_24 = getitem_25 = None
        add_60: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_12: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_60: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_61: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, arg105_1);  mul_60 = arg105_1 = None
        add_61: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, arg106_1);  mul_61 = arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_25: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_61, [0, 3, 1, 2]);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_31: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_25, arg107_1, arg108_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_25 = arg107_1 = arg108_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_62: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_31, 0.5)
        mul_63: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_31, 0.7071067811865476);  convolution_31 = None
        erf_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_62: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_64: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, add_62);  mul_62 = add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_19: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_64, 2)
        sum_10: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_19, [2, 3], True);  pow_19 = None
        pow_20: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_9: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_20, [1], True)
        add_63: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        div_9: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_20, add_63);  pow_20 = add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_65: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, div_9);  div_9 = None
        addcmul_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_18, view_19, mul_65);  view_18 = view_19 = mul_65 = None
        add_64: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_64, addcmul_9);  mul_64 = addcmul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_32: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_64, arg111_1, arg112_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_64 = arg111_1 = arg112_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_65: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_32, add_59);  convolution_32 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_33: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_65, arg113_1, arg114_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg113_1 = arg114_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_26: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_13 = torch.ops.aten.var_mean.correction(permute_26, [3], correction = 0, keepdim = True)
        getitem_26: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_20: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg119_1, [1, -1, 1, 1]);  arg119_1 = None
        view_21: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg120_1, [1, -1, 1, 1]);  arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_13: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_26, getitem_27);  permute_26 = getitem_27 = None
        add_66: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        rsqrt_13: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_66: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_67: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg115_1);  mul_66 = arg115_1 = None
        add_67: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg116_1);  mul_67 = arg116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_27: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_67, [0, 3, 1, 2]);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_34: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_27, arg117_1, arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_27 = arg117_1 = arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_68: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_34, 0.5)
        mul_69: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_34, 0.7071067811865476);  convolution_34 = None
        erf_10: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_68: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_70: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_68);  mul_68 = add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_21: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_70, 2)
        sum_11: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_21, [2, 3], True);  pow_21 = None
        pow_22: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_11, 0.5);  sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_10: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_22, [1], True)
        add_69: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        div_10: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_22, add_69);  pow_22 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_71: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, div_10);  div_10 = None
        addcmul_10: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_20, view_21, mul_71);  view_20 = view_21 = mul_71 = None
        add_70: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_70, addcmul_10);  mul_70 = addcmul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_35: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_70, arg121_1, arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_70 = arg121_1 = arg122_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_71: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_35, add_65);  convolution_35 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_36: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_71, arg123_1, arg124_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg123_1 = arg124_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_28: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_36, [0, 2, 3, 1]);  convolution_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_14 = torch.ops.aten.var_mean.correction(permute_28, [3], correction = 0, keepdim = True)
        getitem_28: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_22: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg129_1, [1, -1, 1, 1]);  arg129_1 = None
        view_23: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg130_1, [1, -1, 1, 1]);  arg130_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_14: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_28, getitem_29);  permute_28 = getitem_29 = None
        add_72: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_14: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_72: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_73: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, arg125_1);  mul_72 = arg125_1 = None
        add_73: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, arg126_1);  mul_73 = arg126_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_29: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_73, [0, 3, 1, 2]);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_37: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(permute_29, arg127_1, arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_29 = arg127_1 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_74: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_37, 0.5)
        mul_75: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_37, 0.7071067811865476);  convolution_37 = None
        erf_11: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_74: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_76: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, add_74);  mul_74 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_23: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_76, 2)
        sum_12: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_23, [2, 3], True);  pow_23 = None
        pow_24: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_12, 0.5);  sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_11: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_24, [1], True)
        add_75: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        div_11: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_24, add_75);  pow_24 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_77: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, div_11);  div_11 = None
        addcmul_11: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_22, view_23, mul_77);  view_22 = view_23 = mul_77 = None
        add_76: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_76, addcmul_11);  mul_76 = addcmul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_38: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_76, arg131_1, arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_76 = arg131_1 = arg132_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_77: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_38, add_71);  convolution_38 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_30: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(add_77, [0, 2, 3, 1]);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_15 = torch.ops.aten.var_mean.correction(permute_30, [3], correction = 0, keepdim = True)
        getitem_30: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_30, getitem_31);  permute_30 = getitem_31 = None
        add_78: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_15: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_78: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_79: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, arg133_1);  mul_78 = arg133_1 = None
        add_79: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, arg134_1);  mul_79 = arg134_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_31: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_79, [0, 3, 1, 2]);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        convolution_39: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(permute_31, arg135_1, arg136_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_31 = arg135_1 = arg136_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_40: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(convolution_39, arg137_1, arg138_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 640);  arg137_1 = arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_32: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_40, [0, 2, 3, 1]);  convolution_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_16 = torch.ops.aten.var_mean.correction(permute_32, [3], correction = 0, keepdim = True)
        getitem_32: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_24: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg143_1, [1, -1, 1, 1]);  arg143_1 = None
        view_25: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg144_1, [1, -1, 1, 1]);  arg144_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_16: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_32, getitem_33);  permute_32 = getitem_33 = None
        add_80: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_16: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_80: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_81: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, arg139_1);  mul_80 = arg139_1 = None
        add_81: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, arg140_1);  mul_81 = arg140_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_33: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.permute.default(add_81, [0, 3, 1, 2]);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_41: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.convolution.default(permute_33, arg141_1, arg142_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_33 = arg141_1 = arg142_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_82: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_41, 0.5)
        mul_83: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_41, 0.7071067811865476);  convolution_41 = None
        erf_12: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_82: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_84: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, add_82);  mul_82 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_25: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_84, 2)
        sum_13: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_25, [2, 3], True);  pow_25 = None
        pow_26: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_13, 0.5);  sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_12: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_26, [1], True)
        add_83: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        div_12: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_26, add_83);  pow_26 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_85: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, div_12);  div_12 = None
        addcmul_12: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.addcmul.default(view_24, view_25, mul_85);  view_24 = view_25 = mul_85 = None
        add_84: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(mul_84, addcmul_12);  mul_84 = addcmul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_42: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(add_84, arg145_1, arg146_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_84 = arg145_1 = arg146_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_85: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.add.Tensor(convolution_42, convolution_39);  convolution_42 = convolution_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convolution_43: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(add_85, arg147_1, arg148_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 640);  arg147_1 = arg148_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_34: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1]);  convolution_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_17 = torch.ops.aten.var_mean.correction(permute_34, [3], correction = 0, keepdim = True)
        getitem_34: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_26: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg153_1, [1, -1, 1, 1]);  arg153_1 = None
        view_27: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg154_1, [1, -1, 1, 1]);  arg154_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_17: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_34, getitem_35);  permute_34 = getitem_35 = None
        add_86: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
        rsqrt_17: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_86: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_87: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, arg149_1);  mul_86 = arg149_1 = None
        add_87: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, arg150_1);  mul_87 = arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_35: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.permute.default(add_87, [0, 3, 1, 2]);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_44: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.convolution.default(permute_35, arg151_1, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_35 = arg151_1 = arg152_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_88: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_44, 0.5)
        mul_89: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476);  convolution_44 = None
        erf_13: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_88: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_90: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, add_88);  mul_88 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_27: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_90, 2)
        sum_14: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_27, [2, 3], True);  pow_27 = None
        pow_28: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_14, 0.5);  sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_13: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_28, [1], True)
        add_89: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        div_13: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_28, add_89);  pow_28 = add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_91: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, div_13);  div_13 = None
        addcmul_13: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.addcmul.default(view_26, view_27, mul_91);  view_26 = view_27 = mul_91 = None
        add_90: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(mul_90, addcmul_13);  mul_90 = addcmul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_45: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(add_90, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_90 = arg155_1 = arg156_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_91: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.add.Tensor(convolution_45, add_85);  convolution_45 = add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_14: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_91, [-1, -2], True);  add_91 = None
        as_strided: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.as_strided.default(mean_14, [128, 640, 1, 1], [640, 1, 640, 640]);  mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_36: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.permute.default(as_strided, [0, 2, 3, 1]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_18 = torch.ops.aten.var_mean.correction(permute_36, [3], correction = 0, keepdim = True)
        getitem_36: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_36, getitem_37);  permute_36 = getitem_37 = None
        add_92: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_18: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_92: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_93: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, arg157_1);  mul_92 = arg157_1 = None
        add_93: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, arg158_1);  mul_93 = arg158_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_37: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.permute.default(add_93, [0, 3, 1, 2]);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:214 in forward, code: x = self.flatten(x)
        view_28: "f32[128, 640][640, 1]cuda:0" = torch.ops.aten.reshape.default(permute_37, [128, 640]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:219 in forward, code: x = self.fc(x)
        permute_38: "f32[640, 1000][1, 640]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm: "f32[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_28, permute_38);  arg160_1 = view_28 = permute_38 = None
        return (addmm,)

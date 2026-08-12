class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", arg1_1: "bf16[768, 3, 16, 16][768, 1, 48, 3]cuda:0", arg2_1: "bf16[768][1]cuda:0", arg3_1: "bf16[1, 1, 768][768, 768, 1]cuda:0", arg4_1: "bf16[768][1]cuda:0", arg5_1: "bf16[768][1]cuda:0", arg6_1: "bf16[768][1]cuda:0", arg7_1: "bf16[768][1]cuda:0", arg8_1: "bf16[768][1]cuda:0", arg9_1: "bf16[768][1]cuda:0", arg10_1: "bf16[2304, 768][768, 1]cuda:0", arg11_1: "bf16[732, 12][12, 1]cuda:0", arg12_1: "i64[197, 197][197, 1]cuda:0", arg13_1: "bf16[768, 768][768, 1]cuda:0", arg14_1: "bf16[768][1]cuda:0", arg15_1: "bf16[768][1]cuda:0", arg16_1: "bf16[768][1]cuda:0", arg17_1: "bf16[768][1]cuda:0", arg18_1: "bf16[3072, 768][768, 1]cuda:0", arg19_1: "bf16[3072][1]cuda:0", arg20_1: "bf16[768, 3072][3072, 1]cuda:0", arg21_1: "bf16[768][1]cuda:0", arg22_1: "bf16[768][1]cuda:0", arg23_1: "bf16[768][1]cuda:0", arg24_1: "bf16[768][1]cuda:0", arg25_1: "bf16[768][1]cuda:0", arg26_1: "bf16[768][1]cuda:0", arg27_1: "bf16[768][1]cuda:0", arg28_1: "bf16[2304, 768][768, 1]cuda:0", arg29_1: "bf16[732, 12][12, 1]cuda:0", arg30_1: "i64[197, 197][197, 1]cuda:0", arg31_1: "bf16[768, 768][768, 1]cuda:0", arg32_1: "bf16[768][1]cuda:0", arg33_1: "bf16[768][1]cuda:0", arg34_1: "bf16[768][1]cuda:0", arg35_1: "bf16[768][1]cuda:0", arg36_1: "bf16[3072, 768][768, 1]cuda:0", arg37_1: "bf16[3072][1]cuda:0", arg38_1: "bf16[768, 3072][3072, 1]cuda:0", arg39_1: "bf16[768][1]cuda:0", arg40_1: "bf16[768][1]cuda:0", arg41_1: "bf16[768][1]cuda:0", arg42_1: "bf16[768][1]cuda:0", arg43_1: "bf16[768][1]cuda:0", arg44_1: "bf16[768][1]cuda:0", arg45_1: "bf16[768][1]cuda:0", arg46_1: "bf16[2304, 768][768, 1]cuda:0", arg47_1: "bf16[732, 12][12, 1]cuda:0", arg48_1: "i64[197, 197][197, 1]cuda:0", arg49_1: "bf16[768, 768][768, 1]cuda:0", arg50_1: "bf16[768][1]cuda:0", arg51_1: "bf16[768][1]cuda:0", arg52_1: "bf16[768][1]cuda:0", arg53_1: "bf16[768][1]cuda:0", arg54_1: "bf16[3072, 768][768, 1]cuda:0", arg55_1: "bf16[3072][1]cuda:0", arg56_1: "bf16[768, 3072][3072, 1]cuda:0", arg57_1: "bf16[768][1]cuda:0", arg58_1: "bf16[768][1]cuda:0", arg59_1: "bf16[768][1]cuda:0", arg60_1: "bf16[768][1]cuda:0", arg61_1: "bf16[768][1]cuda:0", arg62_1: "bf16[768][1]cuda:0", arg63_1: "bf16[768][1]cuda:0", arg64_1: "bf16[2304, 768][768, 1]cuda:0", arg65_1: "bf16[732, 12][12, 1]cuda:0", arg66_1: "i64[197, 197][197, 1]cuda:0", arg67_1: "bf16[768, 768][768, 1]cuda:0", arg68_1: "bf16[768][1]cuda:0", arg69_1: "bf16[768][1]cuda:0", arg70_1: "bf16[768][1]cuda:0", arg71_1: "bf16[768][1]cuda:0", arg72_1: "bf16[3072, 768][768, 1]cuda:0", arg73_1: "bf16[3072][1]cuda:0", arg74_1: "bf16[768, 3072][3072, 1]cuda:0", arg75_1: "bf16[768][1]cuda:0", arg76_1: "bf16[768][1]cuda:0", arg77_1: "bf16[768][1]cuda:0", arg78_1: "bf16[768][1]cuda:0", arg79_1: "bf16[768][1]cuda:0", arg80_1: "bf16[768][1]cuda:0", arg81_1: "bf16[768][1]cuda:0", arg82_1: "bf16[2304, 768][768, 1]cuda:0", arg83_1: "bf16[732, 12][12, 1]cuda:0", arg84_1: "i64[197, 197][197, 1]cuda:0", arg85_1: "bf16[768, 768][768, 1]cuda:0", arg86_1: "bf16[768][1]cuda:0", arg87_1: "bf16[768][1]cuda:0", arg88_1: "bf16[768][1]cuda:0", arg89_1: "bf16[768][1]cuda:0", arg90_1: "bf16[3072, 768][768, 1]cuda:0", arg91_1: "bf16[3072][1]cuda:0", arg92_1: "bf16[768, 3072][3072, 1]cuda:0", arg93_1: "bf16[768][1]cuda:0", arg94_1: "bf16[768][1]cuda:0", arg95_1: "bf16[768][1]cuda:0", arg96_1: "bf16[768][1]cuda:0", arg97_1: "bf16[768][1]cuda:0", arg98_1: "bf16[768][1]cuda:0", arg99_1: "bf16[768][1]cuda:0", arg100_1: "bf16[2304, 768][768, 1]cuda:0", arg101_1: "bf16[732, 12][12, 1]cuda:0", arg102_1: "i64[197, 197][197, 1]cuda:0", arg103_1: "bf16[768, 768][768, 1]cuda:0", arg104_1: "bf16[768][1]cuda:0", arg105_1: "bf16[768][1]cuda:0", arg106_1: "bf16[768][1]cuda:0", arg107_1: "bf16[768][1]cuda:0", arg108_1: "bf16[3072, 768][768, 1]cuda:0", arg109_1: "bf16[3072][1]cuda:0", arg110_1: "bf16[768, 3072][3072, 1]cuda:0", arg111_1: "bf16[768][1]cuda:0", arg112_1: "bf16[768][1]cuda:0", arg113_1: "bf16[768][1]cuda:0", arg114_1: "bf16[768][1]cuda:0", arg115_1: "bf16[768][1]cuda:0", arg116_1: "bf16[768][1]cuda:0", arg117_1: "bf16[768][1]cuda:0", arg118_1: "bf16[2304, 768][768, 1]cuda:0", arg119_1: "bf16[732, 12][12, 1]cuda:0", arg120_1: "i64[197, 197][197, 1]cuda:0", arg121_1: "bf16[768, 768][768, 1]cuda:0", arg122_1: "bf16[768][1]cuda:0", arg123_1: "bf16[768][1]cuda:0", arg124_1: "bf16[768][1]cuda:0", arg125_1: "bf16[768][1]cuda:0", arg126_1: "bf16[3072, 768][768, 1]cuda:0", arg127_1: "bf16[3072][1]cuda:0", arg128_1: "bf16[768, 3072][3072, 1]cuda:0", arg129_1: "bf16[768][1]cuda:0", arg130_1: "bf16[768][1]cuda:0", arg131_1: "bf16[768][1]cuda:0", arg132_1: "bf16[768][1]cuda:0", arg133_1: "bf16[768][1]cuda:0", arg134_1: "bf16[768][1]cuda:0", arg135_1: "bf16[768][1]cuda:0", arg136_1: "bf16[2304, 768][768, 1]cuda:0", arg137_1: "bf16[732, 12][12, 1]cuda:0", arg138_1: "i64[197, 197][197, 1]cuda:0", arg139_1: "bf16[768, 768][768, 1]cuda:0", arg140_1: "bf16[768][1]cuda:0", arg141_1: "bf16[768][1]cuda:0", arg142_1: "bf16[768][1]cuda:0", arg143_1: "bf16[768][1]cuda:0", arg144_1: "bf16[3072, 768][768, 1]cuda:0", arg145_1: "bf16[3072][1]cuda:0", arg146_1: "bf16[768, 3072][3072, 1]cuda:0", arg147_1: "bf16[768][1]cuda:0", arg148_1: "bf16[768][1]cuda:0", arg149_1: "bf16[768][1]cuda:0", arg150_1: "bf16[768][1]cuda:0", arg151_1: "bf16[768][1]cuda:0", arg152_1: "bf16[768][1]cuda:0", arg153_1: "bf16[768][1]cuda:0", arg154_1: "bf16[2304, 768][768, 1]cuda:0", arg155_1: "bf16[732, 12][12, 1]cuda:0", arg156_1: "i64[197, 197][197, 1]cuda:0", arg157_1: "bf16[768, 768][768, 1]cuda:0", arg158_1: "bf16[768][1]cuda:0", arg159_1: "bf16[768][1]cuda:0", arg160_1: "bf16[768][1]cuda:0", arg161_1: "bf16[768][1]cuda:0", arg162_1: "bf16[3072, 768][768, 1]cuda:0", arg163_1: "bf16[3072][1]cuda:0", arg164_1: "bf16[768, 3072][3072, 1]cuda:0", arg165_1: "bf16[768][1]cuda:0", arg166_1: "bf16[768][1]cuda:0", arg167_1: "bf16[768][1]cuda:0", arg168_1: "bf16[768][1]cuda:0", arg169_1: "bf16[768][1]cuda:0", arg170_1: "bf16[768][1]cuda:0", arg171_1: "bf16[768][1]cuda:0", arg172_1: "bf16[2304, 768][768, 1]cuda:0", arg173_1: "bf16[732, 12][12, 1]cuda:0", arg174_1: "i64[197, 197][197, 1]cuda:0", arg175_1: "bf16[768, 768][768, 1]cuda:0", arg176_1: "bf16[768][1]cuda:0", arg177_1: "bf16[768][1]cuda:0", arg178_1: "bf16[768][1]cuda:0", arg179_1: "bf16[768][1]cuda:0", arg180_1: "bf16[3072, 768][768, 1]cuda:0", arg181_1: "bf16[3072][1]cuda:0", arg182_1: "bf16[768, 3072][3072, 1]cuda:0", arg183_1: "bf16[768][1]cuda:0", arg184_1: "bf16[768][1]cuda:0", arg185_1: "bf16[768][1]cuda:0", arg186_1: "bf16[768][1]cuda:0", arg187_1: "bf16[768][1]cuda:0", arg188_1: "bf16[768][1]cuda:0", arg189_1: "bf16[768][1]cuda:0", arg190_1: "bf16[2304, 768][768, 1]cuda:0", arg191_1: "bf16[732, 12][12, 1]cuda:0", arg192_1: "i64[197, 197][197, 1]cuda:0", arg193_1: "bf16[768, 768][768, 1]cuda:0", arg194_1: "bf16[768][1]cuda:0", arg195_1: "bf16[768][1]cuda:0", arg196_1: "bf16[768][1]cuda:0", arg197_1: "bf16[768][1]cuda:0", arg198_1: "bf16[3072, 768][768, 1]cuda:0", arg199_1: "bf16[3072][1]cuda:0", arg200_1: "bf16[768, 3072][3072, 1]cuda:0", arg201_1: "bf16[768][1]cuda:0", arg202_1: "bf16[768][1]cuda:0", arg203_1: "bf16[768][1]cuda:0", arg204_1: "bf16[768][1]cuda:0", arg205_1: "bf16[768][1]cuda:0", arg206_1: "bf16[768][1]cuda:0", arg207_1: "bf16[768][1]cuda:0", arg208_1: "bf16[2304, 768][768, 1]cuda:0", arg209_1: "bf16[732, 12][12, 1]cuda:0", arg210_1: "i64[197, 197][197, 1]cuda:0", arg211_1: "bf16[768, 768][768, 1]cuda:0", arg212_1: "bf16[768][1]cuda:0", arg213_1: "bf16[768][1]cuda:0", arg214_1: "bf16[768][1]cuda:0", arg215_1: "bf16[768][1]cuda:0", arg216_1: "bf16[3072, 768][768, 1]cuda:0", arg217_1: "bf16[3072][1]cuda:0", arg218_1: "bf16[768, 3072][3072, 1]cuda:0", arg219_1: "bf16[768][1]cuda:0", arg220_1: "bf16[768][1]cuda:0", arg221_1: "bf16[768][1]cuda:0", arg222_1: "bf16[1000, 768][768, 1]cuda:0", arg223_1: "bf16[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:787 in forward_features, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        expand: "bf16[128, 1, 768][0, 768, 1]cuda:0" = torch.ops.aten.expand.default(arg3_1, [128, -1, -1]);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convolution: "bf16[128, 768, 14, 14][150528, 1, 10752, 768]cuda:0" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view: "bf16[128, 768, 196][150528, 1, 768]cuda:0" = torch.ops.aten.reshape.default(convolution, [128, 768, 196]);  convolution = None
        permute: "bf16[128, 196, 768][150528, 768, 1]cuda:0" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:787 in forward_features, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        cat: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_1: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg7_1, arg8_1, arg9_1]);  arg7_1 = arg8_1 = arg9_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = arg5_1 = None
        add_1: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        convert_element_type_1: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_1: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [25216, 768]);  convert_element_type_1 = None
        permute_1: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_1, view_1, permute_1);  cat_1 = view_1 = permute_1 = None
        view_2: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [128, 197, 2304]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_3: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_2, [128, 197, 3, 12, -1]);  view_2 = None
        permute_2: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
        getitem_2: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind[0]
        getitem_3: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind[1]
        getitem_4: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_4: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg12_1, [-1]);  arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg11_1, [view_4]);  arg11_1 = view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_5: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index, [197, 197, -1]);  index = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_3: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_5, [2, 0, 1]);  view_5 = None
        clone_1: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_1, 0);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_2, getitem_3, getitem_4, unsqueeze, False);  getitem_2 = getitem_3 = getitem_4 = unsqueeze = None
        getitem_5: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_4: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_6: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_4, [128, 197, 768]);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_7: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [25216, 768]);  view_6 = None
        permute_5: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg14_1, view_7, permute_5);  arg14_1 = view_7 = permute_5 = None
        view_8: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [128, 197, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_2: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg4_1, view_8);  arg4_1 = view_8 = None
        add_2: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(cat, mul_2);  cat = mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_8: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_14: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_1[0]
        getitem_15: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_15);  convert_element_type_8 = getitem_15 = None
        add_3: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_1: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_3: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_4: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, arg16_1);  mul_3 = arg16_1 = None
        add_4: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, arg17_1);  mul_4 = arg17_1 = None
        convert_element_type_9: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_9: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_9, [25216, 768]);  convert_element_type_9 = None
        permute_6: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_2: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg19_1, view_9, permute_6);  arg19_1 = view_9 = permute_6 = None
        view_10: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 197, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_13: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        mul_5: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.5)
        mul_6: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.7071067811865476);  convert_element_type_13 = None
        erf: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_5: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None
        convert_element_type_14: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_11: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_14, [25216, 3072]);  convert_element_type_14 = None
        permute_7: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_3: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg21_1, view_11, permute_7);  arg21_1 = view_11 = permute_7 = None
        view_12: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [128, 197, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_8: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg15_1, view_12);  arg15_1 = view_12 = None
        add_6: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_2, mul_8);  add_2 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_18: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_18, [2], correction = 0, keepdim = True)
        getitem_16: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_2[0]
        getitem_17: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_2: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg25_1, arg26_1, arg27_1]);  arg25_1 = arg26_1 = arg27_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_2: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_18, getitem_17);  convert_element_type_18 = getitem_17 = None
        add_7: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_2: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_9: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_10: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, arg23_1);  mul_9 = arg23_1 = None
        add_8: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, arg24_1);  mul_10 = arg24_1 = None
        convert_element_type_19: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_13: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_19, [25216, 768]);  convert_element_type_19 = None
        permute_8: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_4: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_2, view_13, permute_8);  cat_2 = view_13 = permute_8 = None
        view_14: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [128, 197, 2304]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_15: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [128, 197, 3, 12, -1]);  view_14 = None
        permute_9: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
        getitem_18: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_1[0]
        getitem_19: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_1[1]
        getitem_20: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_16: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg30_1, [-1]);  arg30_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_1: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg29_1, [view_16]);  arg29_1 = view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_17: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_1, [197, 197, -1]);  index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_10: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_17, [2, 0, 1]);  view_17 = None
        clone_5: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_1: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_5, 0);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_18, getitem_19, getitem_20, unsqueeze_1, False);  getitem_18 = getitem_19 = getitem_20 = unsqueeze_1 = None
        getitem_21: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_11: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
        view_18: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_11, [128, 197, 768]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_19: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_18, [25216, 768]);  view_18 = None
        permute_12: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_5: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, view_19, permute_12);  arg32_1 = view_19 = permute_12 = None
        view_20: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [128, 197, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_11: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg22_1, view_20);  arg22_1 = view_20 = None
        add_9: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_6, mul_11);  add_6 = mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_26: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_26, [2], correction = 0, keepdim = True)
        getitem_30: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_3[0]
        getitem_31: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, getitem_31);  convert_element_type_26 = getitem_31 = None
        add_10: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_3: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_12: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_13: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, arg34_1);  mul_12 = arg34_1 = None
        add_11: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, arg35_1);  mul_13 = arg35_1 = None
        convert_element_type_27: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_21: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_27, [25216, 768]);  convert_element_type_27 = None
        permute_13: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_6: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg37_1, view_21, permute_13);  arg37_1 = view_21 = permute_13 = None
        view_22: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 197, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_31: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_22, torch.float32);  view_22 = None
        mul_14: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.5)
        mul_15: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.7071067811865476);  convert_element_type_31 = None
        erf_1: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_12: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, add_12);  mul_14 = add_12 = None
        convert_element_type_32: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_23: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_32, [25216, 3072]);  convert_element_type_32 = None
        permute_14: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_7: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg39_1, view_23, permute_14);  arg39_1 = view_23 = permute_14 = None
        view_24: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [128, 197, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_17: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg33_1, view_24);  arg33_1 = view_24 = None
        add_13: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, mul_17);  add_9 = mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_36: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_36, [2], correction = 0, keepdim = True)
        getitem_32: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_4[0]
        getitem_33: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_3: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg43_1, arg44_1, arg45_1]);  arg43_1 = arg44_1 = arg45_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_4: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_36, getitem_33);  convert_element_type_36 = getitem_33 = None
        add_14: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_4: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_18: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_19: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, arg41_1);  mul_18 = arg41_1 = None
        add_15: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, arg42_1);  mul_19 = arg42_1 = None
        convert_element_type_37: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_25: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_37, [25216, 768]);  convert_element_type_37 = None
        permute_15: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_8: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_3, view_25, permute_15);  cat_3 = view_25 = permute_15 = None
        view_26: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [128, 197, 2304]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_27: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [128, 197, 3, 12, -1]);  view_26 = None
        permute_16: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_27, [2, 0, 3, 1, 4]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_2 = torch.ops.aten.unbind.int(permute_16);  permute_16 = None
        getitem_34: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_2[0]
        getitem_35: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_2[1]
        getitem_36: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_28: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg48_1, [-1]);  arg48_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_2: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg47_1, [view_28]);  arg47_1 = view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_29: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_2, [197, 197, -1]);  index_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_17: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_29, [2, 0, 1]);  view_29 = None
        clone_9: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_2: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_9, 0);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_34, getitem_35, getitem_36, unsqueeze_2, False);  getitem_34 = getitem_35 = getitem_36 = unsqueeze_2 = None
        getitem_37: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_18: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3]);  getitem_37 = None
        view_30: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_18, [128, 197, 768]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_31: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [25216, 768]);  view_30 = None
        permute_19: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_9: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg50_1, view_31, permute_19);  arg50_1 = view_31 = permute_19 = None
        view_32: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [128, 197, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_20: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg40_1, view_32);  arg40_1 = view_32 = None
        add_16: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, mul_20);  add_13 = mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_44: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_44, [2], correction = 0, keepdim = True)
        getitem_46: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_5[0]
        getitem_47: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_44, getitem_47);  convert_element_type_44 = getitem_47 = None
        add_17: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_5: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_21: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_22: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, arg52_1);  mul_21 = arg52_1 = None
        add_18: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, arg53_1);  mul_22 = arg53_1 = None
        convert_element_type_45: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_33: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_45, [25216, 768]);  convert_element_type_45 = None
        permute_20: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_10: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg55_1, view_33, permute_20);  arg55_1 = view_33 = permute_20 = None
        view_34: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 197, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_49: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        mul_23: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_49, 0.5)
        mul_24: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_49, 0.7071067811865476);  convert_element_type_49 = None
        erf_2: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_19: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_25: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, add_19);  mul_23 = add_19 = None
        convert_element_type_50: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_35: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_50, [25216, 3072]);  convert_element_type_50 = None
        permute_21: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_11: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg57_1, view_35, permute_21);  arg57_1 = view_35 = permute_21 = None
        view_36: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [128, 197, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_26: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg51_1, view_36);  arg51_1 = view_36 = None
        add_20: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_16, mul_26);  add_16 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_54: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_54, [2], correction = 0, keepdim = True)
        getitem_48: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_6[0]
        getitem_49: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_4: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg61_1, arg62_1, arg63_1]);  arg61_1 = arg62_1 = arg63_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_6: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_54, getitem_49);  convert_element_type_54 = getitem_49 = None
        add_21: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_6: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_27: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_28: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, arg59_1);  mul_27 = arg59_1 = None
        add_22: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_28, arg60_1);  mul_28 = arg60_1 = None
        convert_element_type_55: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_37: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_55, [25216, 768]);  convert_element_type_55 = None
        permute_22: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_12: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_4, view_37, permute_22);  cat_4 = view_37 = permute_22 = None
        view_38: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [128, 197, 2304]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_39: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_38, [128, 197, 3, 12, -1]);  view_38 = None
        permute_23: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_39, [2, 0, 3, 1, 4]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_3 = torch.ops.aten.unbind.int(permute_23);  permute_23 = None
        getitem_50: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_3[0]
        getitem_51: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_3[1]
        getitem_52: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_40: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg66_1, [-1]);  arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_3: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg65_1, [view_40]);  arg65_1 = view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_41: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_3, [197, 197, -1]);  index_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_24: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_41, [2, 0, 1]);  view_41 = None
        clone_13: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_3: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_13, 0);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_50, getitem_51, getitem_52, unsqueeze_3, False);  getitem_50 = getitem_51 = getitem_52 = unsqueeze_3 = None
        getitem_53: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_25: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        view_42: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_25, [128, 197, 768]);  permute_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_43: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [25216, 768]);  view_42 = None
        permute_26: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_13: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg68_1, view_43, permute_26);  arg68_1 = view_43 = permute_26 = None
        view_44: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [128, 197, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_29: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg58_1, view_44);  arg58_1 = view_44 = None
        add_23: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_20, mul_29);  add_20 = mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_62: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_62, [2], correction = 0, keepdim = True)
        getitem_62: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_7[0]
        getitem_63: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_63);  convert_element_type_62 = getitem_63 = None
        add_24: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_7: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_30: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_31: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, arg70_1);  mul_30 = arg70_1 = None
        add_25: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, arg71_1);  mul_31 = arg71_1 = None
        convert_element_type_63: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_45: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [25216, 768]);  convert_element_type_63 = None
        permute_27: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_14: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg73_1, view_45, permute_27);  arg73_1 = view_45 = permute_27 = None
        view_46: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 197, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_67: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        mul_32: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_33: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476);  convert_element_type_67 = None
        erf_3: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_26: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_34: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, add_26);  mul_32 = add_26 = None
        convert_element_type_68: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_47: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_68, [25216, 3072]);  convert_element_type_68 = None
        permute_28: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_15: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg75_1, view_47, permute_28);  arg75_1 = view_47 = permute_28 = None
        view_48: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [128, 197, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_35: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg69_1, view_48);  arg69_1 = view_48 = None
        add_27: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_23, mul_35);  add_23 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_72: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_64: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_8[0]
        getitem_65: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_5: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg79_1, arg80_1, arg81_1]);  arg79_1 = arg80_1 = arg81_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_8: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_65);  convert_element_type_72 = getitem_65 = None
        add_28: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_8: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_36: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_37: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
        add_29: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
        convert_element_type_73: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_49: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [25216, 768]);  convert_element_type_73 = None
        permute_29: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_16: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_5, view_49, permute_29);  cat_5 = view_49 = permute_29 = None
        view_50: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [128, 197, 2304]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_51: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [128, 197, 3, 12, -1]);  view_50 = None
        permute_30: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_4 = torch.ops.aten.unbind.int(permute_30);  permute_30 = None
        getitem_66: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_4[0]
        getitem_67: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_4[1]
        getitem_68: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_52: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg84_1, [-1]);  arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_4: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg83_1, [view_52]);  arg83_1 = view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_53: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_4, [197, 197, -1]);  index_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_31: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_53, [2, 0, 1]);  view_53 = None
        clone_17: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_4: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_17, 0);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_66, getitem_67, getitem_68, unsqueeze_4, False);  getitem_66 = getitem_67 = getitem_68 = unsqueeze_4 = None
        getitem_69: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_32: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3]);  getitem_69 = None
        view_54: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_32, [128, 197, 768]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_55: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [25216, 768]);  view_54 = None
        permute_33: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_17: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg86_1, view_55, permute_33);  arg86_1 = view_55 = permute_33 = None
        view_56: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [128, 197, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_38: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg76_1, view_56);  arg76_1 = view_56 = None
        add_30: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_27, mul_38);  add_27 = mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_80: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_78: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_9[0]
        getitem_79: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_79);  convert_element_type_80 = getitem_79 = None
        add_31: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
        rsqrt_9: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_39: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_40: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, arg88_1);  mul_39 = arg88_1 = None
        add_32: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_40, arg89_1);  mul_40 = arg89_1 = None
        convert_element_type_81: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_57: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_81, [25216, 768]);  convert_element_type_81 = None
        permute_34: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_18: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg91_1, view_57, permute_34);  arg91_1 = view_57 = permute_34 = None
        view_58: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 197, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_85: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        mul_41: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.5)
        mul_42: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.7071067811865476);  convert_element_type_85 = None
        erf_4: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_33: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_43: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, add_33);  mul_41 = add_33 = None
        convert_element_type_86: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_59: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_86, [25216, 3072]);  convert_element_type_86 = None
        permute_35: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_19: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg93_1, view_59, permute_35);  arg93_1 = view_59 = permute_35 = None
        view_60: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [128, 197, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_44: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg87_1, view_60);  arg87_1 = view_60 = None
        add_34: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_30, mul_44);  add_30 = mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_90: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_80: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_10[0]
        getitem_81: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_6: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg97_1, arg98_1, arg99_1]);  arg97_1 = arg98_1 = arg99_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_10: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_81);  convert_element_type_90 = getitem_81 = None
        add_35: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_10: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_45: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_46: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, arg95_1);  mul_45 = arg95_1 = None
        add_36: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, arg96_1);  mul_46 = arg96_1 = None
        convert_element_type_91: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_61: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [25216, 768]);  convert_element_type_91 = None
        permute_36: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_20: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_6, view_61, permute_36);  cat_6 = view_61 = permute_36 = None
        view_62: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [128, 197, 2304]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_63: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [128, 197, 3, 12, -1]);  view_62 = None
        permute_37: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
        getitem_82: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_5[0]
        getitem_83: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_5[1]
        getitem_84: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_64: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg102_1, [-1]);  arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_5: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg101_1, [view_64]);  arg101_1 = view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_65: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_5, [197, 197, -1]);  index_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_38: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_65, [2, 0, 1]);  view_65 = None
        clone_21: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_5: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_21, 0);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_82, getitem_83, getitem_84, unsqueeze_5, False);  getitem_82 = getitem_83 = getitem_84 = unsqueeze_5 = None
        getitem_85: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_39: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
        view_66: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [128, 197, 768]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_67: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_66, [25216, 768]);  view_66 = None
        permute_40: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_21: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg104_1, view_67, permute_40);  arg104_1 = view_67 = permute_40 = None
        view_68: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [128, 197, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_47: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg94_1, view_68);  arg94_1 = view_68 = None
        add_37: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_34, mul_47);  add_34 = mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_98: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True)
        getitem_94: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_11[0]
        getitem_95: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_95);  convert_element_type_98 = getitem_95 = None
        add_38: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_11: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_48: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_49: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, arg106_1);  mul_48 = arg106_1 = None
        add_39: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, arg107_1);  mul_49 = arg107_1 = None
        convert_element_type_99: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_69: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_99, [25216, 768]);  convert_element_type_99 = None
        permute_41: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_22: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg109_1, view_69, permute_41);  arg109_1 = view_69 = permute_41 = None
        view_70: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 197, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_103: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None
        mul_50: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.5)
        mul_51: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.7071067811865476);  convert_element_type_103 = None
        erf_5: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_40: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_52: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_40);  mul_50 = add_40 = None
        convert_element_type_104: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_71: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_104, [25216, 3072]);  convert_element_type_104 = None
        permute_42: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_23: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg111_1, view_71, permute_42);  arg111_1 = view_71 = permute_42 = None
        view_72: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [128, 197, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_53: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg105_1, view_72);  arg105_1 = view_72 = None
        add_41: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, mul_53);  add_37 = mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_108: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_108, [2], correction = 0, keepdim = True)
        getitem_96: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_12[0]
        getitem_97: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_7: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg115_1, arg116_1, arg117_1]);  arg115_1 = arg116_1 = arg117_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_12: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_108, getitem_97);  convert_element_type_108 = getitem_97 = None
        add_42: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_12: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_54: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_55: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, arg113_1);  mul_54 = arg113_1 = None
        add_43: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, arg114_1);  mul_55 = arg114_1 = None
        convert_element_type_109: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_73: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_109, [25216, 768]);  convert_element_type_109 = None
        permute_43: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_24: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_7, view_73, permute_43);  cat_7 = view_73 = permute_43 = None
        view_74: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [128, 197, 2304]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_75: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [128, 197, 3, 12, -1]);  view_74 = None
        permute_44: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_75, [2, 0, 3, 1, 4]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_6 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
        getitem_98: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_6[0]
        getitem_99: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_6[1]
        getitem_100: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_76: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg120_1, [-1]);  arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_6: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg119_1, [view_76]);  arg119_1 = view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_77: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_6, [197, 197, -1]);  index_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_45: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_77, [2, 0, 1]);  view_77 = None
        clone_25: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_6: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_25, 0);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_98, getitem_99, getitem_100, unsqueeze_6, False);  getitem_98 = getitem_99 = getitem_100 = unsqueeze_6 = None
        getitem_101: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_46: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
        view_78: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_46, [128, 197, 768]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_79: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_78, [25216, 768]);  view_78 = None
        permute_47: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_25: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg122_1, view_79, permute_47);  arg122_1 = view_79 = permute_47 = None
        view_80: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [128, 197, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_56: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg112_1, view_80);  arg112_1 = view_80 = None
        add_44: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_41, mul_56);  add_41 = mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_116: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_116, [2], correction = 0, keepdim = True)
        getitem_110: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_13[0]
        getitem_111: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_116, getitem_111);  convert_element_type_116 = getitem_111 = None
        add_45: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_13: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_57: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_58: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg124_1);  mul_57 = arg124_1 = None
        add_46: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg125_1);  mul_58 = arg125_1 = None
        convert_element_type_117: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_81: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_117, [25216, 768]);  convert_element_type_117 = None
        permute_48: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_26: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg127_1, view_81, permute_48);  arg127_1 = view_81 = permute_48 = None
        view_82: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 197, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_121: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_82, torch.float32);  view_82 = None
        mul_59: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_121, 0.5)
        mul_60: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_121, 0.7071067811865476);  convert_element_type_121 = None
        erf_6: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_47: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_61: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, add_47);  mul_59 = add_47 = None
        convert_element_type_122: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_83: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_122, [25216, 3072]);  convert_element_type_122 = None
        permute_49: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_27: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg129_1, view_83, permute_49);  arg129_1 = view_83 = permute_49 = None
        view_84: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [128, 197, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_62: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg123_1, view_84);  arg123_1 = view_84 = None
        add_48: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_44, mul_62);  add_44 = mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_126: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_126, [2], correction = 0, keepdim = True)
        getitem_112: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_14[0]
        getitem_113: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_8: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg133_1, arg134_1, arg135_1]);  arg133_1 = arg134_1 = arg135_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_14: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_126, getitem_113);  convert_element_type_126 = getitem_113 = None
        add_49: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
        rsqrt_14: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_63: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_64: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, arg131_1);  mul_63 = arg131_1 = None
        add_50: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, arg132_1);  mul_64 = arg132_1 = None
        convert_element_type_127: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_85: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_127, [25216, 768]);  convert_element_type_127 = None
        permute_50: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_28: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_8, view_85, permute_50);  cat_8 = view_85 = permute_50 = None
        view_86: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [128, 197, 2304]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_87: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [128, 197, 3, 12, -1]);  view_86 = None
        permute_51: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_7 = torch.ops.aten.unbind.int(permute_51);  permute_51 = None
        getitem_114: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_7[0]
        getitem_115: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_7[1]
        getitem_116: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_88: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg138_1, [-1]);  arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_7: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg137_1, [view_88]);  arg137_1 = view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_89: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_7, [197, 197, -1]);  index_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_52: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_89, [2, 0, 1]);  view_89 = None
        clone_29: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_7: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_29, 0);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_114, getitem_115, getitem_116, unsqueeze_7, False);  getitem_114 = getitem_115 = getitem_116 = unsqueeze_7 = None
        getitem_117: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_53: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
        view_90: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_53, [128, 197, 768]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_91: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_90, [25216, 768]);  view_90 = None
        permute_54: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_29: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_91, permute_54);  arg140_1 = view_91 = permute_54 = None
        view_92: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [128, 197, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_65: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg130_1, view_92);  arg130_1 = view_92 = None
        add_51: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_48, mul_65);  add_48 = mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_134: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_134, [2], correction = 0, keepdim = True)
        getitem_126: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_15[0]
        getitem_127: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_127);  convert_element_type_134 = getitem_127 = None
        add_52: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
        rsqrt_15: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_66: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_67: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg142_1);  mul_66 = arg142_1 = None
        add_53: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg143_1);  mul_67 = arg143_1 = None
        convert_element_type_135: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_93: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [25216, 768]);  convert_element_type_135 = None
        permute_55: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_30: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg145_1, view_93, permute_55);  arg145_1 = view_93 = permute_55 = None
        view_94: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 197, 3072]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_139: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_94, torch.float32);  view_94 = None
        mul_68: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_69: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_7: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_54: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_70: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_54);  mul_68 = add_54 = None
        convert_element_type_140: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_95: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_140, [25216, 3072]);  convert_element_type_140 = None
        permute_56: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_31: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg147_1, view_95, permute_56);  arg147_1 = view_95 = permute_56 = None
        view_96: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [128, 197, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_71: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg141_1, view_96);  arg141_1 = view_96 = None
        add_55: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_51, mul_71);  add_51 = mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_144: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_128: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_16[0]
        getitem_129: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_9: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg151_1, arg152_1, arg153_1]);  arg151_1 = arg152_1 = arg153_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_16: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_129);  convert_element_type_144 = getitem_129 = None
        add_56: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
        rsqrt_16: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_72: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_73: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, arg149_1);  mul_72 = arg149_1 = None
        add_57: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, arg150_1);  mul_73 = arg150_1 = None
        convert_element_type_145: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_97: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [25216, 768]);  convert_element_type_145 = None
        permute_57: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_32: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_9, view_97, permute_57);  cat_9 = view_97 = permute_57 = None
        view_98: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [128, 197, 2304]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_99: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_98, [128, 197, 3, 12, -1]);  view_98 = None
        permute_58: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_99, [2, 0, 3, 1, 4]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_8 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
        getitem_130: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_8[0]
        getitem_131: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_8[1]
        getitem_132: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_100: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg156_1, [-1]);  arg156_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_8: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg155_1, [view_100]);  arg155_1 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_101: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_8, [197, 197, -1]);  index_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_59: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_101, [2, 0, 1]);  view_101 = None
        clone_33: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_8: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_33, 0);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_130, getitem_131, getitem_132, unsqueeze_8, False);  getitem_130 = getitem_131 = getitem_132 = unsqueeze_8 = None
        getitem_133: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_60: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
        view_102: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_60, [128, 197, 768]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_103: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [25216, 768]);  view_102 = None
        permute_61: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_33: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg158_1, view_103, permute_61);  arg158_1 = view_103 = permute_61 = None
        view_104: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [128, 197, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_74: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg148_1, view_104);  arg148_1 = view_104 = None
        add_58: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_55, mul_74);  add_55 = mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_152: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_152, [2], correction = 0, keepdim = True)
        getitem_142: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_17[0]
        getitem_143: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_152, getitem_143);  convert_element_type_152 = getitem_143 = None
        add_59: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
        rsqrt_17: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_75: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_76: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, arg160_1);  mul_75 = arg160_1 = None
        add_60: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, arg161_1);  mul_76 = arg161_1 = None
        convert_element_type_153: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_60, torch.bfloat16);  add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_105: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_153, [25216, 768]);  convert_element_type_153 = None
        permute_62: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_34: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg163_1, view_105, permute_62);  arg163_1 = view_105 = permute_62 = None
        view_106: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 197, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_157: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_106, torch.float32);  view_106 = None
        mul_77: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.5)
        mul_78: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.7071067811865476);  convert_element_type_157 = None
        erf_8: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_61: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_79: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, add_61);  mul_77 = add_61 = None
        convert_element_type_158: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_107: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_158, [25216, 3072]);  convert_element_type_158 = None
        permute_63: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_35: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg165_1, view_107, permute_63);  arg165_1 = view_107 = permute_63 = None
        view_108: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [128, 197, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_80: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg159_1, view_108);  arg159_1 = view_108 = None
        add_62: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_58, mul_80);  add_58 = mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_162: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_162, [2], correction = 0, keepdim = True)
        getitem_144: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_18[0]
        getitem_145: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_10: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg169_1, arg170_1, arg171_1]);  arg169_1 = arg170_1 = arg171_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_18: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_162, getitem_145);  convert_element_type_162 = getitem_145 = None
        add_63: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
        rsqrt_18: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_81: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_82: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, arg167_1);  mul_81 = arg167_1 = None
        add_64: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, arg168_1);  mul_82 = arg168_1 = None
        convert_element_type_163: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_109: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_163, [25216, 768]);  convert_element_type_163 = None
        permute_64: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_36: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_10, view_109, permute_64);  cat_10 = view_109 = permute_64 = None
        view_110: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [128, 197, 2304]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_111: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_110, [128, 197, 3, 12, -1]);  view_110 = None
        permute_65: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_9 = torch.ops.aten.unbind.int(permute_65);  permute_65 = None
        getitem_146: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_9[0]
        getitem_147: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_9[1]
        getitem_148: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_9[2];  unbind_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_112: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg174_1, [-1]);  arg174_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_9: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg173_1, [view_112]);  arg173_1 = view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_113: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_9, [197, 197, -1]);  index_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_66: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_113, [2, 0, 1]);  view_113 = None
        clone_37: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_9: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_37, 0);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_146, getitem_147, getitem_148, unsqueeze_9, False);  getitem_146 = getitem_147 = getitem_148 = unsqueeze_9 = None
        getitem_149: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_67: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None
        view_114: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_67, [128, 197, 768]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_115: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [25216, 768]);  view_114 = None
        permute_68: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_37: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg176_1, view_115, permute_68);  arg176_1 = view_115 = permute_68 = None
        view_116: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [128, 197, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_83: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg166_1, view_116);  arg166_1 = view_116 = None
        add_65: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62, mul_83);  add_62 = mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_170: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True)
        getitem_158: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_19[0]
        getitem_159: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_159);  convert_element_type_170 = getitem_159 = None
        add_66: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_19: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_84: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_85: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, arg178_1);  mul_84 = arg178_1 = None
        add_67: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, arg179_1);  mul_85 = arg179_1 = None
        convert_element_type_171: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_117: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_171, [25216, 768]);  convert_element_type_171 = None
        permute_69: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_38: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg181_1, view_117, permute_69);  arg181_1 = view_117 = permute_69 = None
        view_118: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 197, 3072]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_175: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        mul_86: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.5)
        mul_87: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.7071067811865476);  convert_element_type_175 = None
        erf_9: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_68: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_88: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, add_68);  mul_86 = add_68 = None
        convert_element_type_176: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_119: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_176, [25216, 3072]);  convert_element_type_176 = None
        permute_70: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_39: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg183_1, view_119, permute_70);  arg183_1 = view_119 = permute_70 = None
        view_120: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [128, 197, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_89: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg177_1, view_120);  arg177_1 = view_120 = None
        add_69: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_65, mul_89);  add_65 = mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_180: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_180, [2], correction = 0, keepdim = True)
        getitem_160: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_20[0]
        getitem_161: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_11: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg187_1, arg188_1, arg189_1]);  arg187_1 = arg188_1 = arg189_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_20: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_180, getitem_161);  convert_element_type_180 = getitem_161 = None
        add_70: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
        rsqrt_20: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_90: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_91: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, arg185_1);  mul_90 = arg185_1 = None
        add_71: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, arg186_1);  mul_91 = arg186_1 = None
        convert_element_type_181: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_121: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_181, [25216, 768]);  convert_element_type_181 = None
        permute_71: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_40: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_11, view_121, permute_71);  cat_11 = view_121 = permute_71 = None
        view_122: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [128, 197, 2304]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_123: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_122, [128, 197, 3, 12, -1]);  view_122 = None
        permute_72: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_123, [2, 0, 3, 1, 4]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_10 = torch.ops.aten.unbind.int(permute_72);  permute_72 = None
        getitem_162: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_10[0]
        getitem_163: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_10[1]
        getitem_164: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_10[2];  unbind_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_124: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg192_1, [-1]);  arg192_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_10: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg191_1, [view_124]);  arg191_1 = view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_125: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_10, [197, 197, -1]);  index_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_73: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_125, [2, 0, 1]);  view_125 = None
        clone_41: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_10: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_41, 0);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_162, getitem_163, getitem_164, unsqueeze_10, False);  getitem_162 = getitem_163 = getitem_164 = unsqueeze_10 = None
        getitem_165: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_74: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
        view_126: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_74, [128, 197, 768]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_127: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_126, [25216, 768]);  view_126 = None
        permute_75: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_41: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg194_1, view_127, permute_75);  arg194_1 = view_127 = permute_75 = None
        view_128: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [128, 197, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_92: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg184_1, view_128);  arg184_1 = view_128 = None
        add_72: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_69, mul_92);  add_69 = mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_188: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_188, [2], correction = 0, keepdim = True)
        getitem_174: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_21[0]
        getitem_175: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, getitem_175);  convert_element_type_188 = getitem_175 = None
        add_73: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
        rsqrt_21: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_93: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_94: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, arg196_1);  mul_93 = arg196_1 = None
        add_74: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_94, arg197_1);  mul_94 = arg197_1 = None
        convert_element_type_189: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_129: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_189, [25216, 768]);  convert_element_type_189 = None
        permute_76: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_42: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg199_1, view_129, permute_76);  arg199_1 = view_129 = permute_76 = None
        view_130: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 197, 3072]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_193: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        mul_95: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_193, 0.5)
        mul_96: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_193, 0.7071067811865476);  convert_element_type_193 = None
        erf_10: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
        add_75: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_97: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, add_75);  mul_95 = add_75 = None
        convert_element_type_194: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_97, torch.bfloat16);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_131: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_194, [25216, 3072]);  convert_element_type_194 = None
        permute_77: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_43: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg201_1, view_131, permute_77);  arg201_1 = view_131 = permute_77 = None
        view_132: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [128, 197, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_98: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg195_1, view_132);  arg195_1 = view_132 = None
        add_76: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_72, mul_98);  add_72 = mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_198: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_198, [2], correction = 0, keepdim = True)
        getitem_176: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_22[0]
        getitem_177: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        cat_12: "bf16[2304][1]cuda:0" = torch.ops.aten.cat.default([arg205_1, arg206_1, arg207_1]);  arg205_1 = arg206_1 = arg207_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_22: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_198, getitem_177);  convert_element_type_198 = getitem_177 = None
        add_77: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_22: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_99: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_100: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, arg203_1);  mul_99 = arg203_1 = None
        add_78: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_100, arg204_1);  mul_100 = arg204_1 = None
        convert_element_type_199: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        view_133: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_199, [25216, 768]);  convert_element_type_199 = None
        permute_78: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_44: "bf16[25216, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(cat_12, view_133, permute_78);  cat_12 = view_133 = permute_78 = None
        view_134: "bf16[128, 197, 2304][453888, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [128, 197, 2304]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_135: "bf16[128, 197, 3, 12, 64][453888, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_134, [128, 197, 3, 12, -1]);  view_134 = None
        permute_79: "bf16[3, 128, 12, 197, 64][768, 453888, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_135, [2, 0, 3, 1, 4]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_11 = torch.ops.aten.unbind.int(permute_79);  permute_79 = None
        getitem_178: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_11[0]
        getitem_179: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_11[1]
        getitem_180: "bf16[128, 12, 197, 64][453888, 64, 2304, 1]cuda:0" = unbind_11[2];  unbind_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_136: "i64[38809][1]cuda:0" = torch.ops.aten.reshape.default(arg210_1, [-1]);  arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_11: "bf16[38809, 12][12, 1]cuda:0" = torch.ops.aten.index.Tensor(arg209_1, [view_136]);  arg209_1 = view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        view_137: "bf16[197, 197, 12][2364, 12, 1]cuda:0" = torch.ops.aten.reshape.default(index_11, [197, 197, -1]);  index_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_80: "bf16[12, 197, 197][1, 2364, 12]cuda:0" = torch.ops.aten.permute.default(view_137, [2, 0, 1]);  view_137 = None
        clone_45: "bf16[12, 197, 197][38809, 197, 1]cuda:0" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_11: "bf16[1, 12, 197, 197][465708, 38809, 197, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_45, 0);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_178, getitem_179, getitem_180, unsqueeze_11, False);  getitem_178 = getitem_179 = getitem_180 = unsqueeze_11 = None
        getitem_181: "bf16[128, 12, 197, 64][151296, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_81: "bf16[128, 197, 12, 64][151296, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
        view_138: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_81, [128, 197, 768]);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        view_139: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_138, [25216, 768]);  view_138 = None
        permute_82: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_45: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg212_1, view_139, permute_82);  arg212_1 = view_139 = permute_82 = None
        view_140: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [128, 197, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_101: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg202_1, view_140);  arg202_1 = view_140 = None
        add_79: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_76, mul_101);  add_76 = mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_206: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_206, [2], correction = 0, keepdim = True)
        getitem_190: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_23[0]
        getitem_191: "f32[128, 197, 1][197, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_23: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_206, getitem_191);  convert_element_type_206 = getitem_191 = None
        add_80: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
        rsqrt_23: "f32[128, 197, 1][197, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_102: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_103: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, arg214_1);  mul_102 = arg214_1 = None
        add_81: "f32[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_103, arg215_1);  mul_103 = arg215_1 = None
        convert_element_type_207: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_141: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_207, [25216, 768]);  convert_element_type_207 = None
        permute_83: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_46: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg217_1, view_141, permute_83);  arg217_1 = view_141 = permute_83 = None
        view_142: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 197, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_211: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_142, torch.float32);  view_142 = None
        mul_104: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.5)
        mul_105: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.7071067811865476);  convert_element_type_211 = None
        erf_11: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_82: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_106: "f32[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, add_82);  mul_104 = add_82 = None
        convert_element_type_212: "bf16[128, 197, 3072][605184, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_106, torch.bfloat16);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_143: "bf16[25216, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_212, [25216, 3072]);  convert_element_type_212 = None
        permute_84: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_47: "bf16[25216, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg219_1, view_143, permute_84);  arg219_1 = view_143 = permute_84 = None
        view_144: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [128, 197, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_107: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg213_1, view_144);  arg213_1 = view_144 = None
        add_83: "bf16[128, 197, 768][151296, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_79, mul_107);  add_79 = mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:812 in forward_head, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        slice_1: "bf16[128, 196, 768][151296, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_83, 1, 1, 9223372036854775807);  add_83 = None
        mean: "bf16[128, 768][768, 1]cuda:0" = torch.ops.aten.mean.dim(slice_1, [1]);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_216: "f32[128, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mean, torch.float32);  mean = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_216, [1], correction = 0, keepdim = True)
        getitem_192: "f32[128, 1][1, 1]cuda:0" = var_mean_24[0]
        getitem_193: "f32[128, 1][1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        sub_24: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_216, getitem_193);  convert_element_type_216 = getitem_193 = None
        add_84: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_24: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_108: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_109: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, arg220_1);  mul_108 = arg220_1 = None
        add_85: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, arg221_1);  mul_109 = arg221_1 = None
        convert_element_type_217: "bf16[128, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:815 in forward_head, code: return x if pre_logits else self.head(x)
        permute_85: "bf16[768, 1000][1, 768]cuda:0" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_48: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg223_1, convert_element_type_217, permute_85);  arg223_1 = convert_element_type_217 = permute_85 = None
        return (addmm_48,)

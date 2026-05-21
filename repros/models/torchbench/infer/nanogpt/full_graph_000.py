class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 64]", arg1_1: "f32[50304, 768]", arg2_1: "f32[1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[2304, 768]", arg6_1: "f32[2304]", arg7_1: "f32[768, 768]", arg8_1: "f32[768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[3072, 768]", arg12_1: "f32[3072]", arg13_1: "f32[768, 3072]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[2304, 768]", arg18_1: "f32[2304]", arg19_1: "f32[768, 768]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[3072, 768]", arg24_1: "f32[3072]", arg25_1: "f32[768, 3072]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[2304, 768]", arg30_1: "f32[2304]", arg31_1: "f32[768, 768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072, 768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[2304, 768]", arg42_1: "f32[2304]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072, 768]", arg48_1: "f32[3072]", arg49_1: "f32[768, 3072]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[2304, 768]", arg54_1: "f32[2304]", arg55_1: "f32[768, 768]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[3072, 768]", arg60_1: "f32[3072]", arg61_1: "f32[768, 3072]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[2304, 768]", arg66_1: "f32[2304]", arg67_1: "f32[768, 768]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[3072, 768]", arg72_1: "f32[3072]", arg73_1: "f32[768, 3072]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[2304, 768]", arg78_1: "f32[2304]", arg79_1: "f32[768, 768]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072, 768]", arg84_1: "f32[3072]", arg85_1: "f32[768, 3072]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[2304, 768]", arg90_1: "f32[2304]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[3072, 768]", arg96_1: "f32[3072]", arg97_1: "f32[768, 3072]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[2304, 768]", arg102_1: "f32[2304]", arg103_1: "f32[768, 768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[3072, 768]", arg108_1: "f32[3072]", arg109_1: "f32[768, 3072]", arg110_1: "f32[768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[2304, 768]", arg114_1: "f32[2304]", arg115_1: "f32[768, 768]", arg116_1: "f32[768]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[3072, 768]", arg120_1: "f32[3072]", arg121_1: "f32[768, 3072]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[2304, 768]", arg126_1: "f32[2304]", arg127_1: "f32[768, 768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072, 768]", arg132_1: "f32[3072]", arg133_1: "f32[768, 3072]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[2304, 768]", arg138_1: "f32[2304]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[3072, 768]", arg144_1: "f32[3072]", arg145_1: "f32[768, 3072]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[768]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:229 in forward, code: tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        embedding: "f32[1, 64, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:224 in forward, code: pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
        iota: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 64]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:230 in forward, code: pos_emb = self.transformer.wpe(
        embedding_1: "f32[1, 64, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = unsqueeze = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:233 in forward, code: x = self.transformer.drop(tok_emb + pos_emb)
        add: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1]" = var_mean[0]
        getitem_1: "f32[1, 64, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        add_1: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_2: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view: "f32[64, 768]" = torch.ops.aten.reshape.default(add_2, [64, 768]);  add_2 = None
        permute: "f32[768, 2304]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
        view_1: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm, [1, 64, 2304]);  addmm = None
        split = torch.ops.aten.split.Tensor(view_1, 768, 2);  view_1 = None
        getitem_2: "f32[1, 64, 768]" = split[0]
        getitem_3: "f32[1, 64, 768]" = split[1]
        getitem_4: "f32[1, 64, 768]" = split[2];  split = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_3: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [1, 64, 12, 64]);  getitem_2 = None
        permute_2: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_2: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [1, 64, 12, 64]);  getitem_3 = None
        permute_1: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_4: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [1, 64, 12, 64]);  getitem_4 = None
        permute_3: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute_1, permute_3, None, False, 0.0, True);  permute_2 = permute_1 = permute_3 = None
        getitem_5: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_4: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_5: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_4, [1, 64, 768]);  permute_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_6: "f32[64, 768]" = torch.ops.aten.reshape.default(view_5, [64, 768]);  view_5 = None
        permute_5: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "f32[64, 768]" = torch.ops.aten.addmm.default(arg8_1, view_6, permute_5);  arg8_1 = view_6 = permute_5 = None
        view_7: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 64, 768]);  addmm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_3: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add, view_7);  add = view_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_9: "f32[1, 64, 1]" = var_mean_1[0]
        getitem_10: "f32[1, 64, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_10);  getitem_10 = None
        add_4: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_2: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_5: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_8: "f32[64, 768]" = torch.ops.aten.reshape.default(add_5, [64, 768]);  add_5 = None
        permute_6: "f32[768, 3072]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg12_1, view_8, permute_6);  arg12_1 = view_8 = permute_6 = None
        view_9: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_2, [1, 64, 3072]);  addmm_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_4: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_9, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_1: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_9, 3.0)
        mul_5: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_6: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_9, mul_5);  view_9 = mul_5 = None
        mul_6: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
        tanh: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_7: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_7: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_10: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_7, [64, 3072]);  mul_7 = None
        permute_7: "f32[3072, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_3: "f32[64, 768]" = torch.ops.aten.addmm.default(arg14_1, view_10, permute_7);  arg14_1 = view_10 = permute_7 = None
        view_11: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 64, 768]);  addmm_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_8: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_3, view_11);  add_3 = view_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_11: "f32[1, 64, 1]" = var_mean_2[0]
        getitem_12: "f32[1, 64, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_12);  getitem_12 = None
        add_9: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_8: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_10: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_12: "f32[64, 768]" = torch.ops.aten.reshape.default(add_10, [64, 768]);  add_10 = None
        permute_8: "f32[768, 2304]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_4: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg18_1, view_12, permute_8);  arg18_1 = view_12 = permute_8 = None
        view_13: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_4, [1, 64, 2304]);  addmm_4 = None
        split_1 = torch.ops.aten.split.Tensor(view_13, 768, 2);  view_13 = None
        getitem_13: "f32[1, 64, 768]" = split_1[0]
        getitem_14: "f32[1, 64, 768]" = split_1[1]
        getitem_15: "f32[1, 64, 768]" = split_1[2];  split_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_15: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_13, [1, 64, 12, 64]);  getitem_13 = None
        permute_10: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_14: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_14, [1, 64, 12, 64]);  getitem_14 = None
        permute_9: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_16: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_15, [1, 64, 12, 64]);  getitem_15 = None
        permute_11: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_9, permute_11, None, False, 0.0, True);  permute_10 = permute_9 = permute_11 = None
        getitem_16: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_12: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        view_17: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_12, [1, 64, 768]);  permute_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_18: "f32[64, 768]" = torch.ops.aten.reshape.default(view_17, [64, 768]);  view_17 = None
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_5: "f32[64, 768]" = torch.ops.aten.addmm.default(arg20_1, view_18, permute_13);  arg20_1 = view_18 = permute_13 = None
        view_19: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 64, 768]);  addmm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_11: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_8, view_19);  add_8 = view_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 64, 1]" = var_mean_3[0]
        getitem_21: "f32[1, 64, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_21);  getitem_21 = None
        add_12: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_10: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_13: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_20: "f32[64, 768]" = torch.ops.aten.reshape.default(add_13, [64, 768]);  add_13 = None
        permute_14: "f32[768, 3072]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_6: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg24_1, view_20, permute_14);  arg24_1 = view_20 = permute_14 = None
        view_21: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_6, [1, 64, 3072]);  addmm_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_12: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_21, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_2: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
        mul_13: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_14: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_21, mul_13);  view_21 = mul_13 = None
        mul_14: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh_1: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_15: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_15: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_22: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_15, [64, 3072]);  mul_15 = None
        permute_15: "f32[3072, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_7: "f32[64, 768]" = torch.ops.aten.addmm.default(arg26_1, view_22, permute_15);  arg26_1 = view_22 = permute_15 = None
        view_23: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 64, 768]);  addmm_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_16: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_11, view_23);  add_11 = view_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 64, 1]" = var_mean_4[0]
        getitem_23: "f32[1, 64, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_23);  getitem_23 = None
        add_17: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_16: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_18: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_24: "f32[64, 768]" = torch.ops.aten.reshape.default(add_18, [64, 768]);  add_18 = None
        permute_16: "f32[768, 2304]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_8: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg30_1, view_24, permute_16);  arg30_1 = view_24 = permute_16 = None
        view_25: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_8, [1, 64, 2304]);  addmm_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_25, 768, 2);  view_25 = None
        getitem_24: "f32[1, 64, 768]" = split_2[0]
        getitem_25: "f32[1, 64, 768]" = split_2[1]
        getitem_26: "f32[1, 64, 768]" = split_2[2];  split_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_27: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_24, [1, 64, 12, 64]);  getitem_24 = None
        permute_18: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_26: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_25, [1, 64, 12, 64]);  getitem_25 = None
        permute_17: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_28: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_26, [1, 64, 12, 64]);  getitem_26 = None
        permute_19: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_17, permute_19, None, False, 0.0, True);  permute_18 = permute_17 = permute_19 = None
        getitem_27: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_20: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        view_29: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_20, [1, 64, 768]);  permute_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_30: "f32[64, 768]" = torch.ops.aten.reshape.default(view_29, [64, 768]);  view_29 = None
        permute_21: "f32[768, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_9: "f32[64, 768]" = torch.ops.aten.addmm.default(arg32_1, view_30, permute_21);  arg32_1 = view_30 = permute_21 = None
        view_31: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 64, 768]);  addmm_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_19: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_16, view_31);  add_16 = view_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_31: "f32[1, 64, 1]" = var_mean_5[0]
        getitem_32: "f32[1, 64, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_32);  getitem_32 = None
        add_20: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_18: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_19: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_21: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_32: "f32[64, 768]" = torch.ops.aten.reshape.default(add_21, [64, 768]);  add_21 = None
        permute_22: "f32[768, 3072]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_10: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg36_1, view_32, permute_22);  arg36_1 = view_32 = permute_22 = None
        view_33: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 64, 3072]);  addmm_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_20: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_3: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_33, 3.0)
        mul_21: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_22: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_33, mul_21);  view_33 = mul_21 = None
        mul_22: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
        tanh_2: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_23: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_23: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_34: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_23, [64, 3072]);  mul_23 = None
        permute_23: "f32[3072, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_11: "f32[64, 768]" = torch.ops.aten.addmm.default(arg38_1, view_34, permute_23);  arg38_1 = view_34 = permute_23 = None
        view_35: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 64, 768]);  addmm_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_24: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_19, view_35);  add_19 = view_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_33: "f32[1, 64, 1]" = var_mean_6[0]
        getitem_34: "f32[1, 64, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_34);  getitem_34 = None
        add_25: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_24: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_25: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_26: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_36: "f32[64, 768]" = torch.ops.aten.reshape.default(add_26, [64, 768]);  add_26 = None
        permute_24: "f32[768, 2304]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_12: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg42_1, view_36, permute_24);  arg42_1 = view_36 = permute_24 = None
        view_37: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_12, [1, 64, 2304]);  addmm_12 = None
        split_3 = torch.ops.aten.split.Tensor(view_37, 768, 2);  view_37 = None
        getitem_35: "f32[1, 64, 768]" = split_3[0]
        getitem_36: "f32[1, 64, 768]" = split_3[1]
        getitem_37: "f32[1, 64, 768]" = split_3[2];  split_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_39: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_35, [1, 64, 12, 64]);  getitem_35 = None
        permute_26: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_38: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_36, [1, 64, 12, 64]);  getitem_36 = None
        permute_25: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_40: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_37, [1, 64, 12, 64]);  getitem_37 = None
        permute_27: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_25, permute_27, None, False, 0.0, True);  permute_26 = permute_25 = permute_27 = None
        getitem_38: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_28: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        view_41: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_28, [1, 64, 768]);  permute_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_42: "f32[64, 768]" = torch.ops.aten.reshape.default(view_41, [64, 768]);  view_41 = None
        permute_29: "f32[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_13: "f32[64, 768]" = torch.ops.aten.addmm.default(arg44_1, view_42, permute_29);  arg44_1 = view_42 = permute_29 = None
        view_43: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 64, 768]);  addmm_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_27: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_24, view_43);  add_24 = view_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 64, 1]" = var_mean_7[0]
        getitem_43: "f32[1, 64, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_43);  getitem_43 = None
        add_28: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_26: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_27: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_29: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_44: "f32[64, 768]" = torch.ops.aten.reshape.default(add_29, [64, 768]);  add_29 = None
        permute_30: "f32[768, 3072]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_14: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg48_1, view_44, permute_30);  arg48_1 = view_44 = permute_30 = None
        view_45: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_14, [1, 64, 3072]);  addmm_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_28: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_4: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_45, 3.0)
        mul_29: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_30: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_45, mul_29);  view_45 = mul_29 = None
        mul_30: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
        tanh_3: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_31: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_31: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_46: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_31, [64, 3072]);  mul_31 = None
        permute_31: "f32[3072, 768]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_15: "f32[64, 768]" = torch.ops.aten.addmm.default(arg50_1, view_46, permute_31);  arg50_1 = view_46 = permute_31 = None
        view_47: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 64, 768]);  addmm_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_32: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_27, view_47);  add_27 = view_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 64, 1]" = var_mean_8[0]
        getitem_45: "f32[1, 64, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_45);  getitem_45 = None
        add_33: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_32: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_33: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_34: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_48: "f32[64, 768]" = torch.ops.aten.reshape.default(add_34, [64, 768]);  add_34 = None
        permute_32: "f32[768, 2304]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_16: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg54_1, view_48, permute_32);  arg54_1 = view_48 = permute_32 = None
        view_49: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_16, [1, 64, 2304]);  addmm_16 = None
        split_4 = torch.ops.aten.split.Tensor(view_49, 768, 2);  view_49 = None
        getitem_46: "f32[1, 64, 768]" = split_4[0]
        getitem_47: "f32[1, 64, 768]" = split_4[1]
        getitem_48: "f32[1, 64, 768]" = split_4[2];  split_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_51: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_46, [1, 64, 12, 64]);  getitem_46 = None
        permute_34: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_50: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_47, [1, 64, 12, 64]);  getitem_47 = None
        permute_33: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_52: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_48, [1, 64, 12, 64]);  getitem_48 = None
        permute_35: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_33, permute_35, None, False, 0.0, True);  permute_34 = permute_33 = permute_35 = None
        getitem_49: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_36: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        view_53: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_36, [1, 64, 768]);  permute_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_54: "f32[64, 768]" = torch.ops.aten.reshape.default(view_53, [64, 768]);  view_53 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_17: "f32[64, 768]" = torch.ops.aten.addmm.default(arg56_1, view_54, permute_37);  arg56_1 = view_54 = permute_37 = None
        view_55: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 64, 768]);  addmm_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_35: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_32, view_55);  add_32 = view_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_53: "f32[1, 64, 1]" = var_mean_9[0]
        getitem_54: "f32[1, 64, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_54);  getitem_54 = None
        add_36: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_34: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_35: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_37: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_56: "f32[64, 768]" = torch.ops.aten.reshape.default(add_37, [64, 768]);  add_37 = None
        permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_18: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg60_1, view_56, permute_38);  arg60_1 = view_56 = permute_38 = None
        view_57: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_18, [1, 64, 3072]);  addmm_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_36: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_57, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_5: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_57, 3.0)
        mul_37: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_38: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_57, mul_37);  view_57 = mul_37 = None
        mul_38: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
        tanh_4: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_39: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_39: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_58: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_39, [64, 3072]);  mul_39 = None
        permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_19: "f32[64, 768]" = torch.ops.aten.addmm.default(arg62_1, view_58, permute_39);  arg62_1 = view_58 = permute_39 = None
        view_59: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 64, 768]);  addmm_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_40: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_35, view_59);  add_35 = view_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_55: "f32[1, 64, 1]" = var_mean_10[0]
        getitem_56: "f32[1, 64, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_56);  getitem_56 = None
        add_41: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_40: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_41: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_42: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_60: "f32[64, 768]" = torch.ops.aten.reshape.default(add_42, [64, 768]);  add_42 = None
        permute_40: "f32[768, 2304]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_20: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg66_1, view_60, permute_40);  arg66_1 = view_60 = permute_40 = None
        view_61: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_20, [1, 64, 2304]);  addmm_20 = None
        split_5 = torch.ops.aten.split.Tensor(view_61, 768, 2);  view_61 = None
        getitem_57: "f32[1, 64, 768]" = split_5[0]
        getitem_58: "f32[1, 64, 768]" = split_5[1]
        getitem_59: "f32[1, 64, 768]" = split_5[2];  split_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_63: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_57, [1, 64, 12, 64]);  getitem_57 = None
        permute_42: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_62: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, [1, 64, 12, 64]);  getitem_58 = None
        permute_41: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_64: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_59, [1, 64, 12, 64]);  getitem_59 = None
        permute_43: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_41, permute_43, None, False, 0.0, True);  permute_42 = permute_41 = permute_43 = None
        getitem_60: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_44: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        view_65: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_44, [1, 64, 768]);  permute_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_66: "f32[64, 768]" = torch.ops.aten.reshape.default(view_65, [64, 768]);  view_65 = None
        permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_21: "f32[64, 768]" = torch.ops.aten.addmm.default(arg68_1, view_66, permute_45);  arg68_1 = view_66 = permute_45 = None
        view_67: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 64, 768]);  addmm_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_43: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_40, view_67);  add_40 = view_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 64, 1]" = var_mean_11[0]
        getitem_65: "f32[1, 64, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_65);  getitem_65 = None
        add_44: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_42: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_43: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_45: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_68: "f32[64, 768]" = torch.ops.aten.reshape.default(add_45, [64, 768]);  add_45 = None
        permute_46: "f32[768, 3072]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_22: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg72_1, view_68, permute_46);  arg72_1 = view_68 = permute_46 = None
        view_69: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 64, 3072]);  addmm_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_44: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_6: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_69, 3.0)
        mul_45: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_46: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_69, mul_45);  view_69 = mul_45 = None
        mul_46: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_47: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_47: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_70: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_47, [64, 3072]);  mul_47 = None
        permute_47: "f32[3072, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_23: "f32[64, 768]" = torch.ops.aten.addmm.default(arg74_1, view_70, permute_47);  arg74_1 = view_70 = permute_47 = None
        view_71: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 64, 768]);  addmm_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_48: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_43, view_71);  add_43 = view_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_66: "f32[1, 64, 1]" = var_mean_12[0]
        getitem_67: "f32[1, 64, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_12: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_67);  getitem_67 = None
        add_49: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_48: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_49: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_50: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_72: "f32[64, 768]" = torch.ops.aten.reshape.default(add_50, [64, 768]);  add_50 = None
        permute_48: "f32[768, 2304]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_24: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg78_1, view_72, permute_48);  arg78_1 = view_72 = permute_48 = None
        view_73: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_24, [1, 64, 2304]);  addmm_24 = None
        split_6 = torch.ops.aten.split.Tensor(view_73, 768, 2);  view_73 = None
        getitem_68: "f32[1, 64, 768]" = split_6[0]
        getitem_69: "f32[1, 64, 768]" = split_6[1]
        getitem_70: "f32[1, 64, 768]" = split_6[2];  split_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_75: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_68, [1, 64, 12, 64]);  getitem_68 = None
        permute_50: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_74: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_69, [1, 64, 12, 64]);  getitem_69 = None
        permute_49: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_76: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_70, [1, 64, 12, 64]);  getitem_70 = None
        permute_51: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_50, permute_49, permute_51, None, False, 0.0, True);  permute_50 = permute_49 = permute_51 = None
        getitem_71: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_52: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        view_77: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_52, [1, 64, 768]);  permute_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_78: "f32[64, 768]" = torch.ops.aten.reshape.default(view_77, [64, 768]);  view_77 = None
        permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_25: "f32[64, 768]" = torch.ops.aten.addmm.default(arg80_1, view_78, permute_53);  arg80_1 = view_78 = permute_53 = None
        view_79: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 64, 768]);  addmm_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_51: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_48, view_79);  add_48 = view_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_75: "f32[1, 64, 1]" = var_mean_13[0]
        getitem_76: "f32[1, 64, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_76);  getitem_76 = None
        add_52: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_50: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_51: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg81_1);  mul_50 = arg81_1 = None
        add_53: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_51, arg82_1);  mul_51 = arg82_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_80: "f32[64, 768]" = torch.ops.aten.reshape.default(add_53, [64, 768]);  add_53 = None
        permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_26: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg84_1, view_80, permute_54);  arg84_1 = view_80 = permute_54 = None
        view_81: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_26, [1, 64, 3072]);  addmm_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_52: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_81, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_7: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_81, 3.0)
        mul_53: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_54: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_81, mul_53);  view_81 = mul_53 = None
        mul_54: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
        tanh_6: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_55: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_55: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_82: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_55, [64, 3072]);  mul_55 = None
        permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_27: "f32[64, 768]" = torch.ops.aten.addmm.default(arg86_1, view_82, permute_55);  arg86_1 = view_82 = permute_55 = None
        view_83: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 64, 768]);  addmm_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_56: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_51, view_83);  add_51 = view_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_77: "f32[1, 64, 1]" = var_mean_14[0]
        getitem_78: "f32[1, 64, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_78);  getitem_78 = None
        add_57: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_56: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_57: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg87_1);  mul_56 = arg87_1 = None
        add_58: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_57, arg88_1);  mul_57 = arg88_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_84: "f32[64, 768]" = torch.ops.aten.reshape.default(add_58, [64, 768]);  add_58 = None
        permute_56: "f32[768, 2304]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_28: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg90_1, view_84, permute_56);  arg90_1 = view_84 = permute_56 = None
        view_85: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_28, [1, 64, 2304]);  addmm_28 = None
        split_7 = torch.ops.aten.split.Tensor(view_85, 768, 2);  view_85 = None
        getitem_79: "f32[1, 64, 768]" = split_7[0]
        getitem_80: "f32[1, 64, 768]" = split_7[1]
        getitem_81: "f32[1, 64, 768]" = split_7[2];  split_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_87: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_79, [1, 64, 12, 64]);  getitem_79 = None
        permute_58: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_86: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_80, [1, 64, 12, 64]);  getitem_80 = None
        permute_57: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_88: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_81, [1, 64, 12, 64]);  getitem_81 = None
        permute_59: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_58, permute_57, permute_59, None, False, 0.0, True);  permute_58 = permute_57 = permute_59 = None
        getitem_82: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_60: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_89: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_60, [1, 64, 768]);  permute_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_90: "f32[64, 768]" = torch.ops.aten.reshape.default(view_89, [64, 768]);  view_89 = None
        permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_29: "f32[64, 768]" = torch.ops.aten.addmm.default(arg92_1, view_90, permute_61);  arg92_1 = view_90 = permute_61 = None
        view_91: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 64, 768]);  addmm_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_59: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_56, view_91);  add_56 = view_91 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_86: "f32[1, 64, 1]" = var_mean_15[0]
        getitem_87: "f32[1, 64, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_87);  getitem_87 = None
        add_60: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_58: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_59: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg93_1);  mul_58 = arg93_1 = None
        add_61: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_59, arg94_1);  mul_59 = arg94_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_92: "f32[64, 768]" = torch.ops.aten.reshape.default(add_61, [64, 768]);  add_61 = None
        permute_62: "f32[768, 3072]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_30: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg96_1, view_92, permute_62);  arg96_1 = view_92 = permute_62 = None
        view_93: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_30, [1, 64, 3072]);  addmm_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_60: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_8: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_93, 3.0)
        mul_61: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_62: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_93, mul_61);  view_93 = mul_61 = None
        mul_62: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_7: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_63: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_63: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_94: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_63, [64, 3072]);  mul_63 = None
        permute_63: "f32[3072, 768]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_31: "f32[64, 768]" = torch.ops.aten.addmm.default(arg98_1, view_94, permute_63);  arg98_1 = view_94 = permute_63 = None
        view_95: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_31, [1, 64, 768]);  addmm_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_64: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_59, view_95);  add_59 = view_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 64, 1]" = var_mean_16[0]
        getitem_89: "f32[1, 64, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_16: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_89);  getitem_89 = None
        add_65: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_64: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_65: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg99_1);  mul_64 = arg99_1 = None
        add_66: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_65, arg100_1);  mul_65 = arg100_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_96: "f32[64, 768]" = torch.ops.aten.reshape.default(add_66, [64, 768]);  add_66 = None
        permute_64: "f32[768, 2304]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_32: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg102_1, view_96, permute_64);  arg102_1 = view_96 = permute_64 = None
        view_97: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_32, [1, 64, 2304]);  addmm_32 = None
        split_8 = torch.ops.aten.split.Tensor(view_97, 768, 2);  view_97 = None
        getitem_90: "f32[1, 64, 768]" = split_8[0]
        getitem_91: "f32[1, 64, 768]" = split_8[1]
        getitem_92: "f32[1, 64, 768]" = split_8[2];  split_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_99: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_90, [1, 64, 12, 64]);  getitem_90 = None
        permute_66: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_98: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_91, [1, 64, 12, 64]);  getitem_91 = None
        permute_65: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_100: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_92, [1, 64, 12, 64]);  getitem_92 = None
        permute_67: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_66, permute_65, permute_67, None, False, 0.0, True);  permute_66 = permute_65 = permute_67 = None
        getitem_93: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_68: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        view_101: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_68, [1, 64, 768]);  permute_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_102: "f32[64, 768]" = torch.ops.aten.reshape.default(view_101, [64, 768]);  view_101 = None
        permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_33: "f32[64, 768]" = torch.ops.aten.addmm.default(arg104_1, view_102, permute_69);  arg104_1 = view_102 = permute_69 = None
        view_103: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 64, 768]);  addmm_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_67: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_64, view_103);  add_64 = view_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_97: "f32[1, 64, 1]" = var_mean_17[0]
        getitem_98: "f32[1, 64, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_98);  getitem_98 = None
        add_68: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_66: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_67: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg105_1);  mul_66 = arg105_1 = None
        add_69: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_67, arg106_1);  mul_67 = arg106_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_104: "f32[64, 768]" = torch.ops.aten.reshape.default(add_69, [64, 768]);  add_69 = None
        permute_70: "f32[768, 3072]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_34: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg108_1, view_104, permute_70);  arg108_1 = view_104 = permute_70 = None
        view_105: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 64, 3072]);  addmm_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_68: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_9: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_105, 3.0)
        mul_69: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_70: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_105, mul_69);  view_105 = mul_69 = None
        mul_70: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_8: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_71: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_71: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_106: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_71, [64, 3072]);  mul_71 = None
        permute_71: "f32[3072, 768]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_35: "f32[64, 768]" = torch.ops.aten.addmm.default(arg110_1, view_106, permute_71);  arg110_1 = view_106 = permute_71 = None
        view_107: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 64, 768]);  addmm_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_72: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_67, view_107);  add_67 = view_107 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_99: "f32[1, 64, 1]" = var_mean_18[0]
        getitem_100: "f32[1, 64, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_100);  getitem_100 = None
        add_73: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_72: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_73: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg111_1);  mul_72 = arg111_1 = None
        add_74: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_73, arg112_1);  mul_73 = arg112_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_108: "f32[64, 768]" = torch.ops.aten.reshape.default(add_74, [64, 768]);  add_74 = None
        permute_72: "f32[768, 2304]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_36: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg114_1, view_108, permute_72);  arg114_1 = view_108 = permute_72 = None
        view_109: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_36, [1, 64, 2304]);  addmm_36 = None
        split_9 = torch.ops.aten.split.Tensor(view_109, 768, 2);  view_109 = None
        getitem_101: "f32[1, 64, 768]" = split_9[0]
        getitem_102: "f32[1, 64, 768]" = split_9[1]
        getitem_103: "f32[1, 64, 768]" = split_9[2];  split_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_111: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_101, [1, 64, 12, 64]);  getitem_101 = None
        permute_74: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_110: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_102, [1, 64, 12, 64]);  getitem_102 = None
        permute_73: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_112: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_103, [1, 64, 12, 64]);  getitem_103 = None
        permute_75: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_74, permute_73, permute_75, None, False, 0.0, True);  permute_74 = permute_73 = permute_75 = None
        getitem_104: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_76: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        view_113: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_76, [1, 64, 768]);  permute_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_114: "f32[64, 768]" = torch.ops.aten.reshape.default(view_113, [64, 768]);  view_113 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_37: "f32[64, 768]" = torch.ops.aten.addmm.default(arg116_1, view_114, permute_77);  arg116_1 = view_114 = permute_77 = None
        view_115: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_37, [1, 64, 768]);  addmm_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_75: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_72, view_115);  add_72 = view_115 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_108: "f32[1, 64, 1]" = var_mean_19[0]
        getitem_109: "f32[1, 64, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_109);  getitem_109 = None
        add_76: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_74: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_75: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg117_1);  mul_74 = arg117_1 = None
        add_77: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_75, arg118_1);  mul_75 = arg118_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_116: "f32[64, 768]" = torch.ops.aten.reshape.default(add_77, [64, 768]);  add_77 = None
        permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_38: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg120_1, view_116, permute_78);  arg120_1 = view_116 = permute_78 = None
        view_117: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_38, [1, 64, 3072]);  addmm_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_76: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_117, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_10: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_117, 3.0)
        mul_77: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_78: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_117, mul_77);  view_117 = mul_77 = None
        mul_78: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
        tanh_9: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_79: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_79: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_118: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_79, [64, 3072]);  mul_79 = None
        permute_79: "f32[3072, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_39: "f32[64, 768]" = torch.ops.aten.addmm.default(arg122_1, view_118, permute_79);  arg122_1 = view_118 = permute_79 = None
        view_119: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_39, [1, 64, 768]);  addmm_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_80: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_75, view_119);  add_75 = view_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_110: "f32[1, 64, 1]" = var_mean_20[0]
        getitem_111: "f32[1, 64, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_20: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_111);  getitem_111 = None
        add_81: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        mul_80: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_81: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg123_1);  mul_80 = arg123_1 = None
        add_82: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_81, arg124_1);  mul_81 = arg124_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_120: "f32[64, 768]" = torch.ops.aten.reshape.default(add_82, [64, 768]);  add_82 = None
        permute_80: "f32[768, 2304]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_40: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg126_1, view_120, permute_80);  arg126_1 = view_120 = permute_80 = None
        view_121: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_40, [1, 64, 2304]);  addmm_40 = None
        split_10 = torch.ops.aten.split.Tensor(view_121, 768, 2);  view_121 = None
        getitem_112: "f32[1, 64, 768]" = split_10[0]
        getitem_113: "f32[1, 64, 768]" = split_10[1]
        getitem_114: "f32[1, 64, 768]" = split_10[2];  split_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_123: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_112, [1, 64, 12, 64]);  getitem_112 = None
        permute_82: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_122: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_113, [1, 64, 12, 64]);  getitem_113 = None
        permute_81: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_124: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_114, [1, 64, 12, 64]);  getitem_114 = None
        permute_83: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_82, permute_81, permute_83, None, False, 0.0, True);  permute_82 = permute_81 = permute_83 = None
        getitem_115: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_84: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        view_125: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_84, [1, 64, 768]);  permute_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_126: "f32[64, 768]" = torch.ops.aten.reshape.default(view_125, [64, 768]);  view_125 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_41: "f32[64, 768]" = torch.ops.aten.addmm.default(arg128_1, view_126, permute_85);  arg128_1 = view_126 = permute_85 = None
        view_127: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_41, [1, 64, 768]);  addmm_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_83: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_80, view_127);  add_80 = view_127 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_119: "f32[1, 64, 1]" = var_mean_21[0]
        getitem_120: "f32[1, 64, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_120);  getitem_120 = None
        add_84: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_82: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_83: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg129_1);  mul_82 = arg129_1 = None
        add_85: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_83, arg130_1);  mul_83 = arg130_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_128: "f32[64, 768]" = torch.ops.aten.reshape.default(add_85, [64, 768]);  add_85 = None
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_42: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg132_1, view_128, permute_86);  arg132_1 = view_128 = permute_86 = None
        view_129: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_42, [1, 64, 3072]);  addmm_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_84: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_11: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_129, 3.0)
        mul_85: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_86: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_129, mul_85);  view_129 = mul_85 = None
        mul_86: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
        tanh_10: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_87: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_87: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_130: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_87, [64, 3072]);  mul_87 = None
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_43: "f32[64, 768]" = torch.ops.aten.addmm.default(arg134_1, view_130, permute_87);  arg134_1 = view_130 = permute_87 = None
        view_131: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_43, [1, 64, 768]);  addmm_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_88: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_83, view_131);  add_83 = view_131 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_121: "f32[1, 64, 1]" = var_mean_22[0]
        getitem_122: "f32[1, 64, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_22: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_122);  getitem_122 = None
        add_89: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_88: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_89: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg135_1);  mul_88 = arg135_1 = None
        add_90: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_89, arg136_1);  mul_89 = arg136_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        view_132: "f32[64, 768]" = torch.ops.aten.reshape.default(add_90, [64, 768]);  add_90 = None
        permute_88: "f32[768, 2304]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_44: "f32[64, 2304]" = torch.ops.aten.addmm.default(arg138_1, view_132, permute_88);  arg138_1 = view_132 = permute_88 = None
        view_133: "f32[1, 64, 2304]" = torch.ops.aten.reshape.default(addmm_44, [1, 64, 2304]);  addmm_44 = None
        split_11 = torch.ops.aten.split.Tensor(view_133, 768, 2);  view_133 = None
        getitem_123: "f32[1, 64, 768]" = split_11[0]
        getitem_124: "f32[1, 64, 768]" = split_11[1]
        getitem_125: "f32[1, 64, 768]" = split_11[2];  split_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:87 in forward, code: q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        view_135: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_123, [1, 64, 12, 64]);  getitem_123 = None
        permute_90: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:84 in forward, code: k = k.view(B, T, self.n_head, C // self.n_head).transpose(
        view_134: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_124, [1, 64, 12, 64]);  getitem_124 = None
        permute_89: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:90 in forward, code: v = v.view(B, T, self.n_head, C // self.n_head).transpose(
        view_136: "f32[1, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_125, [1, 64, 12, 64]);  getitem_125 = None
        permute_91: "f32[1, 12, 64, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:97 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_90, permute_89, permute_91, None, False, 0.0, True);  permute_90 = permute_89 = permute_91 = None
        getitem_126: "f32[1, 12, 64, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:113 in forward, code: y.transpose(1, 2).contiguous().view(B, T, C)
        permute_92: "f32[1, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        view_137: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(permute_92, [1, 64, 768]);  permute_92 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_138: "f32[64, 768]" = torch.ops.aten.reshape.default(view_137, [64, 768]);  view_137 = None
        permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_45: "f32[64, 768]" = torch.ops.aten.addmm.default(arg140_1, view_138, permute_93);  arg140_1 = view_138 = permute_93 = None
        view_139: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_45, [1, 64, 768]);  addmm_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_91: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_88, view_139);  add_88 = view_139 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_130: "f32[1, 64, 1]" = var_mean_23[0]
        getitem_131: "f32[1, 64, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_23: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_131);  getitem_131 = None
        add_92: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_90: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_91: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg141_1);  mul_90 = arg141_1 = None
        add_93: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_91, arg142_1);  mul_91 = arg142_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        view_140: "f32[64, 768]" = torch.ops.aten.reshape.default(add_93, [64, 768]);  add_93 = None
        permute_94: "f32[768, 3072]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_46: "f32[64, 3072]" = torch.ops.aten.addmm.default(arg144_1, view_140, permute_94);  arg144_1 = view_140 = permute_94 = None
        view_141: "f32[1, 64, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 64, 3072]);  addmm_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_92: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:30 in new_gelu, code: + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        pow_12: "f32[1, 64, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_141, 3.0)
        mul_93: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_94: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(view_141, mul_93);  view_141 = mul_93 = None
        mul_94: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
        tanh_11: "f32[1, 64, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:29 in new_gelu, code: 1.0
        add_95: "f32[1, 64, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:26 in new_gelu, code: 0.5
        mul_95: "f32[1, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        view_142: "f32[64, 3072]" = torch.ops.aten.reshape.default(mul_95, [64, 3072]);  mul_95 = None
        permute_95: "f32[3072, 768]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_47: "f32[64, 768]" = torch.ops.aten.addmm.default(arg146_1, view_142, permute_95);  arg146_1 = view_142 = permute_95 = None
        view_143: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_47, [1, 64, 768]);  addmm_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:146 in forward, code: x = x + self.mlp(self.ln_2(x))
        add_96: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_91, view_143);  add_91 = view_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_132: "f32[1, 64, 1]" = var_mean_24[0]
        getitem_133: "f32[1, 64, 1]" = var_mean_24[1];  var_mean_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:240 in forward, code: x[:, [-1], :]
        _tensor_constant0: "i64[1]" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        sub_24: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_133);  add_96 = getitem_133 = None
        add_97: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_96: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_97: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg147_1);  mul_96 = arg147_1 = None
        add_98: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_97, arg148_1);  mul_97 = arg148_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:240 in forward, code: x[:, [-1], :]
        full_default: "i64[1]" = torch.ops.aten.full.default([1], -1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index: "f32[1, 1, 768]" = torch.ops.aten.index.Tensor(add_98, [None, full_default]);  add_98 = full_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:239 in forward, code: logits = self.lm_head(
        view_144: "f32[1, 768]" = torch.ops.aten.reshape.default(index, [1, 768]);  index = None
        permute_96: "f32[768, 50304]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "f32[1, 50304]" = torch.ops.aten.mm.default(view_144, permute_96);  view_144 = permute_96 = None
        view_145: "f32[1, 1, 50304]" = torch.ops.aten.reshape.default(mm, [1, 1, 50304]);  mm = None
        return (view_145,)
